
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.image import preprocess_inputs
import open3d as o3d
import numpy as np
from PIL import Image
import copy
import gtsam
import re
import math

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_IMAGE_PATH = "/home/tong/recordings/rocks/rocks1"
MAX_FILTER_DISTANCE = 15.0 
VOXEL_SIZE = 0.01 # 10cm voxel size for fusing.


KEYFRAME_STEP = 60       # The gap between keyframes (e.g., 10, 20, 30)
END_FRAME = 500     # The last frame number you want to process
NUM_VIEWS_PER_BATCH = 5  

# --- Robust mapping parameters ---
MIN_NODE_TRANSLATION = 0.15  # meters; skip very small motions
MIN_NODE_ROTATION = math.radians(5.0)  # radians; skip tiny rotations
MIN_LOOP_FITNESS = 0.45  # ICP fitness threshold for accepting loop closures
MAX_LOOP_RMSE = 0.04     # Maximum allowable ICP inlier RMSE for loop closures
MAX_LOOP_CANDIDATES = 10 # Limit ICP attempts per new node
OPTIMIZATION_MAX_PASSES = 3
OPTIMIZATION_ERROR_EPS = 1e-3
# ----------------------------------------

# --- GTSAM CONFIGURATION  ---
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
)

LOOP_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
)

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
)

# How far (in meters) to search for old nodes to link to
LOCAL_SEARCH_RADIUS = VOXEL_SIZE  # e.g. 0.1 * 10 = 1.0 meter
LOCAL_WINDOW_SIZE = 5

# --- Model and Intrinsics ---
model = MapAnything.from_pretrained("facebook/map-anything").to(device)

transform_matrix = np.array([
    [1,  0,  0,  0],  # Stays the same
    [0, -1,  0,  0],  # Inverts Y
    [0,  0, -1,  0],  
    [0,  0,  0,  1]
])

intrinsics = np.array([
    [980.21, 0.0, 825.18],
    [0.0, 980.21, 627.85],
    [0.0,    0.0,   1.0 ],
], dtype=np.float32)

# --- Global Lists ---
geometries = [] # Stores final *visualization* geometries
poses = []      # Stores final *optimized* global poses (as np.array)
global_pcd_list = [] # Stores *raw* global point clouds (for ICP)

# --- NEW: Global Graph Objects ---
graph = gtsam.NonlinearFactorGraph()
initial_estimates = gtsam.Values()
pcd_database = {}



def np_to_gtsam(pose_np):
    """Convert numpy 4x4 matrix to gtsam.Pose3"""
    rot_np = pose_np[0:3, 0:3]
    trans_np = pose_np[0:3, 3]
    return gtsam.Pose3(gtsam.Rot3(rot_np), gtsam.Point3(trans_np))

def gtsam_to_np(pose_gtsam):
    """Convert gtsam.Pose3 to numpy 4x4 matrix"""
    pose_np = np.eye(4)
    pose_np[0:3, 0:3] = pose_gtsam.rotation().matrix()
    pose_np[0:3, 3] = pose_gtsam.translation()
    return pose_np.astype(np.float32)


def rotation_angle_between(pose_a, pose_b):
    """Compute the angular difference between two SE3 poses."""
    relative_rot = pose_a[0:3, 0:3].T @ pose_b[0:3, 0:3]
    trace_val = np.trace(relative_rot)
    cos_theta = max(min((trace_val - 1.0) * 0.5, 1.0), -1.0)
    return abs(math.acos(cos_theta))

def run_icp(source_pcd, target_pcd_fused, initial_guess_transform):
    """Helper to run ICP and return the result"""
    # We need a source cloud that is *already moved* by the VO guess
    source_cloud_transformed = copy.deepcopy(source_pcd)
    source_cloud_transformed.transform(initial_guess_transform)
    
    # ICP needs normals for point-to-plane
    target_pcd_fused.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE, max_nn=30))
    source_cloud_transformed.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE, max_nn=30))
    
    # Run ICP
    icp_result = o3d.pipelines.registration.registration_icp(
        source_cloud_transformed, 
        target_pcd_fused, 
        VOXEL_SIZE, # Use voxel size * 2 as correspondence threshold
        np.identity(4),  # We use identity matrix because we already transformed the source
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return icp_result



def initial_map(preds):
    """Processes the very first batch AND initializes the GTSAM graph."""
    global graph, initial_estimates, poses, pcd_database

    print("--- Initializing Map and Graph ---")

    # --- NEW: Add the "anchor" to the graph ---
    # We add a "prior" to the first pose (Node 0) to fix it at the origin.
    # This "nails" the map in place.
    first_pose_gtsam = gtsam.Pose3() # This is an identity matrix
    graph.add(gtsam.PriorFactorPose3(0, first_pose_gtsam, PRIOR_NOISE))
    initial_estimates.insert(0, first_pose_gtsam)
    
    # Also add it to our Python lists
    first_pose_np = gtsam_to_np(first_pose_gtsam)
    poses.append(first_pose_np)
    
    last_pose_np = first_pose_np

    for i, pred in enumerate(preds):
        # Get raw points, colors, and the pose
        points_world = pred["pts3d"].squeeze().cpu().numpy().reshape(-1, 3)
        colors = pred["img_no_norm"].squeeze().cpu().numpy().reshape(-1, 3)
        camera_pose_np = pred["camera_poses"].squeeze().cpu().numpy().astype(np.float32)

        # --- Filtering---
        camera_origin = camera_pose_np[:3, 3]
        distances = np.linalg.norm(points_world - camera_origin, axis=1)
        mask = distances <= MAX_FILTER_DISTANCE
        filtered_points = points_world[mask]
        filtered_colors = colors[mask]
        
        # Create raw point cloud for ICP and storage
        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(filtered_points)
        pcd_raw.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        # Store in our lists
        global_pcd_list.append(pcd_raw)
        pcd_database[i] = pcd_raw # Store in the full database
        
        # --- NEW: Add nodes and odometry edges to the graph ---
        if i > 0: # We already added Node 0
            # Add the new node
            pose_gtsam = np_to_gtsam(camera_pose_np)
            initial_estimates.insert(i, pose_gtsam)
            poses.append(camera_pose_np) # Add to our numpy list
            
            # Add the odometry constraint (edge)
            relative_pose = np_to_gtsam(np.linalg.inv(last_pose_np) @ camera_pose_np)
            graph.add(gtsam.BetweenFactorPose3(i-1, i, relative_pose, ODOMETRY_NOISE))
            
        last_pose_np = camera_pose_np # Update for next loop

        # --- Visualization (no change) ---
        pcd_vis = copy.deepcopy(pcd_raw)
        pcd_vis.transform(transform_matrix)
        
        camera_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5) 
        camera_frame_vis.transform(camera_pose_np)
        camera_frame_vis.transform(transform_matrix)
        
        geometries.append(pcd_vis)
        geometries.append(camera_frame_vis)

    # --- NEW: Run an initial optimization ---
    print("Running initial graph optimization...")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
    result = optimizer.optimize()
    
    # Update all our poses with the optimized results
    initial_estimates = result
    for i in range(len(poses)):
        poses[i] = gtsam_to_np(result.atPose3(i))
    
    print(f"Initial map created. Graph has {graph.size()} factors.")



def view_process(batch):
    """Prepares a batch of images for the model."""
    views = []
    for i, img in enumerate(batch):
        if i == len(batch)-1:
            # This is the new frame
            views.append({
                "img": np.array(Image.open(img).convert("RGB")),
                "intrinsics": intrinsics,
                "is_metric_scale": torch.tensor([True], device=device),
            })
        else:
            # These are "anchor" frames. We give their *most recent optimized poses*.
            anchor_pose_index = len(poses) - 4 + i
            views.append({
                "img": np.array(Image.open(img).convert("RGB")),
                "intrinsics": intrinsics,
                "camera_poses": poses[anchor_pose_index], 
                "is_metric_scale": torch.tensor([True], device=device),
            })
    processed_views = preprocess_inputs(views)

    return processed_views




def mapping(preds):
    """Processes the last frame, performs local graph optimization, and adds to map."""
    global graph, initial_estimates, poses, pcd_database

    pred = preds[len(preds)-1]
    
    # --- 1. Get New Data (in local frame) ---
    points_local = pred["pts3d_cam"].squeeze().cpu().numpy().reshape(-1, 3)
    colors = pred["img_no_norm"].squeeze().cpu().numpy().reshape(-1, 3)
    camera_pose_relative = pred["camera_poses"].squeeze().cpu().numpy().astype(np.float32)

    # Filter
    camera_origin_local = np.array([0.0, 0.0, 0.0])
    distances = np.linalg.norm(points_local - camera_origin_local, axis=1)
    mask = distances <= MAX_FILTER_DISTANCE
    filtered_points = points_local[mask]
    filtered_colors = colors[mask]

    # Create the *Source* Point Cloud
    pcd_local_source = o3d.geometry.PointCloud()
    pcd_local_source.points = o3d.utility.Vector3dVector(filtered_points)
    pcd_local_source.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    if not pcd_local_source.has_points():
        print("Warning: No points in new cloud, skipping frame.")
        return

    # --- 2. Get VO "Initial Guess" Pose ---
    # This is the pose of the *start* of the window (e.g., frame 240)
    window_start_pose_index = max(0, len(poses) - 4)
    global_pose_of_window_start = poses[window_start_pose_index]

    # This is our VO guess for the new frame (e.g., frame 300)
    camera_pose_global_guess = (global_pose_of_window_start @ camera_pose_relative).astype(np.float32)

    # --- 3. Motion gating to suppress near-duplicate nodes ---
    prev_node_pose = poses[-1]
    translation_delta = np.linalg.norm(camera_pose_global_guess[0:3, 3] - prev_node_pose[0:3, 3])
    rotation_delta = rotation_angle_between(prev_node_pose, camera_pose_global_guess)
    if translation_delta < MIN_NODE_TRANSLATION and rotation_delta < MIN_NODE_ROTATION:
        print(
            f"Skipping frame: Δtrans={translation_delta:.3f} m, Δrot={math.degrees(rotation_delta):.2f}° "
            "below thresholds."
        )
        return

    # --- 4. Add New Node and Odometry Edge ---
    new_node_id = len(poses)
    prev_node_id = new_node_id - 1

    # Add the new pose *guess* to our list of estimates
    initial_estimates.insert(new_node_id, np_to_gtsam(camera_pose_global_guess))

    # Calculate the relative pose between them (this is our odometry constraint)
    # T_rel = T_prev_inv * T_new
    relative_pose = np.linalg.inv(prev_node_pose) @ camera_pose_global_guess

    # Add the odometry edge to the graph
    graph.add(gtsam.BetweenFactorPose3(prev_node_id, new_node_id,
                                       np_to_gtsam(relative_pose), ODOMETRY_NOISE))
    
    # --- 4. Add "Web" Constraints (Local Optimization) ---
    print(f"--- Node {new_node_id}: Searching for local constraints ---")
    
    nearby_node_ids = []
    
    # A. Find nearby nodes from the sliding window
    for i in range(1, LOCAL_WINDOW_SIZE + 1):
        if new_node_id - i >= 0:
            nearby_node_ids.append(new_node_id - i)
            
    # B. Find nearby nodes by *distance*
    new_pose_translation = camera_pose_global_guess[0:3, 3]
    for i in range(new_node_id - LOCAL_WINDOW_SIZE): # Search all nodes *before* the window
        old_pose_translation = poses[i][0:3, 3]
        dist = np.linalg.norm(new_pose_translation - old_pose_translation)
        if dist < LOCAL_SEARCH_RADIUS:
            nearby_node_ids.append(i)
            
    # Get unique IDs
    nearby_node_ids = sorted(list(set(nearby_node_ids)))
    
    if not nearby_node_ids:
        print("Warning: No nearby nodes found for local map.")

    loop_closure_added = False
    best_pose = camera_pose_global_guess
    best_score = -np.inf

    # B. Run ICP against each nearby node and add loop closures when valid
    for node_id in nearby_node_ids[:MAX_LOOP_CANDIDATES]:
        if node_id not in pcd_database:
            continue

        target_cloud = copy.deepcopy(pcd_database[node_id])
        target_cloud.transform(poses[node_id])
        target_cloud = target_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)
        if not target_cloud.has_points():
            continue

        icp_result = run_icp(pcd_local_source, target_cloud, camera_pose_global_guess)
        fitness = icp_result.fitness
        rmse = getattr(icp_result, "inlier_rmse", None)
        if fitness < MIN_LOOP_FITNESS or (rmse is not None and rmse > MAX_LOOP_RMSE):
            continue

        corrected_global_pose = (icp_result.transformation @ camera_pose_global_guess).astype(np.float32)
        relative_pose = np.linalg.inv(poses[node_id]) @ corrected_global_pose
        graph.add(
            gtsam.BetweenFactorPose3(node_id, new_node_id, np_to_gtsam(relative_pose), LOOP_NOISE)
        )
        loop_closure_added = True

        score = fitness - (rmse if rmse is not None else 0.0)
        if score > best_score:
            best_score = score
            best_pose = corrected_global_pose

        print(
            f"Loop closure added between nodes {node_id} and {new_node_id}: "
            f"fitness={fitness:.3f}, rmse={rmse:.4f}"
        )

    if loop_closure_added:
        initial_estimates.update(new_node_id, np_to_gtsam(best_pose))
    else:
        print("No reliable loop closure found for this node.")

    # --- 5. Optimize the Graph! ---
    optimization_values = initial_estimates
    prev_error = graph.error(optimization_values)
    for pass_idx in range(OPTIMIZATION_MAX_PASSES):
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, optimization_values)
        result = optimizer.optimize()
        new_error = graph.error(result)
        print(
            f"Graph optimization pass {pass_idx + 1}: error {prev_error:.6f} -> {new_error:.6f}"
        )
        optimization_values = result
        if abs(prev_error - new_error) < OPTIMIZATION_ERROR_EPS:
            print("Converged early; stopping optimization loop.")
            break
        prev_error = new_error

    # --- 6. Update All Poses ---
    # Update our "database" of pose estimates for the *next* loop
    initial_estimates = optimization_values
    
    # Update our Python list of 'poses'
    poses = []
    for i in range(new_node_id + 1):
        poses.append(gtsam_to_np(optimization_values.atPose3(i)))

    # --- 7. Store Geometry ---
    camera_pose_global_final = poses[new_node_id] # Get the *optimized* pose
    
    # Store the local cloud (untransformed) in the database
    pcd_database[new_node_id] = pcd_local_source

    # Create the global cloud for fusing, using the *optimized* pose
    pcd_global = copy.deepcopy(pcd_local_source)
    pcd_global.transform(camera_pose_global_final)
    global_pcd_list.append(pcd_global)
    
    # Create and store the *visualization* geometries
    pcd_vis = copy.deepcopy(pcd_global)
    pcd_vis.transform(transform_matrix)
    
    camera_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5) 
    camera_frame_vis.transform(camera_pose_global_final)
    camera_frame_vis.transform(transform_matrix)
    
    geometries.append(pcd_vis)
    geometries.append(camera_frame_vis)




IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _natural_key(path):
    filename = os.path.basename(path)
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", filename)]


image_files = [
    os.path.join(BASE_IMAGE_PATH, fname)
    for fname in os.listdir(BASE_IMAGE_PATH)
    if fname.lower().endswith(IMAGE_EXTENSIONS)
]

if not image_files:
    raise RuntimeError(f"No images found in directory: {BASE_IMAGE_PATH}")

image_files = sorted(image_files, key=_natural_key)
total_frames = len(image_files)

window_span = (NUM_VIEWS_PER_BATCH - 1) * KEYFRAME_STEP

print(f"Processing initial batch (indices 0 to {window_span} with step {KEYFRAME_STEP})...")
batch1 = []
for i in range(NUM_VIEWS_PER_BATCH):
    frame_index = i * KEYFRAME_STEP
    if frame_index >= total_frames:
        raise RuntimeError(
            f"Not enough images ({total_frames}) for initial batch with KEYFRAME_STEP={KEYFRAME_STEP} and NUM_VIEWS_PER_BATCH={NUM_VIEWS_PER_BATCH}."
        )
    batch1.append(image_files[frame_index])

# --- This part is the same ---
views1 = []
for img in batch1:
    views1.append({
        "img": np.array(Image.open(img).convert("RGB")),
        "intrinsics": intrinsics,
        "is_metric_scale": torch.tensor([True], device=device),
    })

views1_1 = preprocess_inputs(views1)
preds1 = model.infer(
    views1_1,
    memory_efficient_inference=False,
    use_amp=True,
    amp_dtype="bf16",
    apply_mask=False,
    mask_edges=False,
    apply_confidence_mask=False,
    confidence_percentile=10,
)

initial_map(preds1) # This now initializes the graph
print(f"Initial map created. Current poses: {len(poses)}")


# ---
# 2. MAPPING LOOP (NOW DYNAMIC)
# ---
print(f"Starting batch processing loop (step {KEYFRAME_STEP})...")
max_frame_index = min(END_FRAME, total_frames - 1)
if max_frame_index < window_span:
    print("Insufficient frames for batch loop; skipping incremental mapping.")
    max_frame_index = window_span

start_frame = KEYFRAME_STEP
stop_frame = (max_frame_index - window_span) + 1
step = KEYFRAME_STEP

for i in range(start_frame, stop_frame, step):
    print(f"\n===== Processing batch: indices {i} to {i + window_span} =====")
    
    # 1. Create the batch paths dynamically
    batch = []
    for j in range(NUM_VIEWS_PER_BATCH):
        frame_index = i + (j * KEYFRAME_STEP)
        if frame_index >= total_frames:
            break
        batch.append(image_files[frame_index])

    if len(batch) < NUM_VIEWS_PER_BATCH:
        print("Warning: Reached end of image list before completing batch; stopping loop.")
        break

    # 2. Process views
    # This will correctly use poses[-4:] inside the function
    views = view_process(batch)
    
    # 3. Run inference
    preds = model.infer(
        views,
        memory_efficient_inference=False,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=False,
        mask_edges=False,
        apply_confidence_mask=False,
        confidence_percentile=10,
    )
    
    # 4. Map the new frame (frame i + window_span)
    mapping(preds)

print("All batch processing complete.")
print(f"Total poses calculated: {len(poses)}") 


# ---
# 3. FUSING AND VISUALIZATION (This part is now much cleaner)
# ---
print("All batches processed. Fusing point clouds...")

# Create one big point cloud to hold everything
global_pcd_raw = o3d.geometry.PointCloud()

# Create a list for your final geometries (map + poses)
scripps924_2 = [] # Using your variable name

# Build the final cloud from the raw, *OPTIMIZED*, unfused list
for pcd in global_pcd_list:
    global_pcd_raw += pcd

# Get the (already transformed) camera poses from the 'geometries' list
for geo in geometries:
    if isinstance(geo, o3d.geometry.TriangleMesh):
        scripps924_2.append(geo)

print(f"Total points before fusing: {len(global_pcd_raw.points)}")

# Now, merge all the overlapping points into voxels
# This will work perfectly because the poses have been optimized!
fused_pcd_raw = global_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
if fused_pcd_raw.has_points():
    filtered_pcd, indices = fused_pcd_raw.remove_radius_outlier(
        nb_points=8,
        radius=VOXEL_SIZE * 6.0,
    )
    if len(indices) > 0:
        fused_pcd_raw = filtered_pcd

print(f"Total points after fusing: {len(fused_pcd_raw.points)}")

# Apply the visualization transform *only* to the final fused cloud
fused_pcd_vis = fused_pcd_raw.transform(transform_matrix)

# Add the single fused map to your final geometry list
scripps924_2.insert(0, fused_pcd_vis) # Insert at the front

# --- Add Camera Path ---
print("Creating camera trajectory path...")
path_points = [p[0:3, 3] for p in poses] # Get from our *optimized* poses list
path_lines = [[i, i + 1] for i in range(len(path_points) - 1)]
path_colors = [[1, 0, 0] for _ in range(len(path_lines))] # Red path
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(path_points),
    lines=o3d.utility.Vector2iVector(path_lines),
)
line_set.colors = o3d.utility.Vector3dVector(path_colors)
line_set.transform(transform_matrix)
scripps924_2.append(line_set)


o3d.visualization.draw_geometries(scripps924_2)
