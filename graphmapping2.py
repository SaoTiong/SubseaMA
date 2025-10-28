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

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_IMAGE_PATH = "/home/tong/recordings/rocks/rocks1"
MAX_FILTER_DISTANCE = 15.0 
VOXEL_SIZE = 0.01 # 10cm voxel size for fusing.


KEYFRAME_STEP = 60       # The gap between keyframes (e.g., 10, 20, 30)
END_FRAME = 500     # The last frame number you want to process
NUM_VIEWS_PER_BATCH = 5  
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
# global_pcd_list = [] # Stores *raw* global point clouds (for ICP)

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


def initial_map(preds):
    
    global graph, initial_estimates, poses, pcd_database
    global PRIOR_NOISE, ODOMETRY_NOISE, MAX_FILTER_DISTANCE, transform_matrix

    print("--- Initializing Map and Graph ---")

    # We'll fill these and then run LM once at the end.
    # Node indexing: node_id = index in preds (0..len(preds)-1)

    # -------------------------------------------------
    # 1. Add node 0 (world origin) with a prior factor
    # -------------------------------------------------
    # World definition:
    # We CHOOSE the first camera in this first batch to be the world frame.
    # i.e. pose[0] = identity 4x4.
    world_pose_np = np.eye(4, dtype=np.float32)  # 4x4
    world_pose_gtsam = gtsam.Pose3()             # identity Pose3

    # Add prior on node 0 so the graph doesn't float away
    graph.add(gtsam.PriorFactorPose3(0, world_pose_gtsam, PRIOR_NOISE))
    initial_estimates.insert(0, world_pose_gtsam)

    # poses[] is our python list of global poses (4x4)
    poses.clear()
    poses.append(world_pose_np)

    # We'll keep track of the "previous" pose while we add more nodes
    last_pose_np = world_pose_np

    # -------------------------------------------------
    # 2. Process each view in the first batch
    # -------------------------------------------------
    for node_id, pred in enumerate(preds):
        # Extract this view's geometry
        # IMPORTANT: use pts3d_cam (camera-local coords),
        # not pts3d (which may already be world-ish in the model's internal frame).
        pts_local = pred["pts3d_cam"].squeeze().cpu().numpy().reshape(-1, 3)
        colors = pred["img_no_norm"].squeeze().cpu().numpy().reshape(-1, 3)

        # Pose of this camera relative to the FIRST camera in the batch
        cam_pose_rel = pred["camera_poses"].squeeze().cpu().numpy().astype(np.float32)

        # Filter by distance to this camera to drop far junk
        dists = np.linalg.norm(pts_local, axis=1)
        keep_mask = dists <= MAX_FILTER_DISTANCE
        pts_local_filt = pts_local[keep_mask]
        colors_filt = colors[keep_mask]

        # Build this node's local-cloud point cloud (camera frame)
        pcd_local = o3d.geometry.PointCloud()
        pcd_local.points = o3d.utility.Vector3dVector(pts_local_filt)
        pcd_local.colors = o3d.utility.Vector3dVector(colors_filt)

        # Save that *local* cloud in the database for later ICP / final fusion
        pcd_database[node_id] = pcd_local

        # Compute this node's initial global pose guess in numpy 4x4
        # node 0: identity
        # node >0: cam_pose_rel is T_cam0->cam_i
        # world frame IS cam0, so world_T_cam_i = cam0_T_cam_i = cam_pose_rel
        if node_id == 0:
            node_global_pose_np = world_pose_np
        else:
            node_global_pose_np = cam_pose_rel

        # Push this node's pose guess into initial_estimates for GTSAM
        if node_id > 0:
            initial_estimates.insert(node_id, np_to_gtsam(node_global_pose_np))

        # Add odometry-like edges between consecutive nodes to connect the batch internally
        # Even though every pose is relative to cam0, we also connect i-1 -> i
        # so the graph "knows" about intra-batch consistency
        if node_id > 0:
            prev_global_pose_np = poses[node_id - 1]  # last appended pose (before optimization)
            rel_pose_np = np.linalg.inv(prev_global_pose_np) @ node_global_pose_np
            graph.add(
                gtsam.BetweenFactorPose3(
                    node_id - 1,
                    node_id,
                    np_to_gtsam(rel_pose_np),
                    ODOMETRY_NOISE,
                )
            )

        # Stash pose guess in our python list `poses`
        if node_id == 0:
            # node 0 already appended above
            pass
        else:
            poses.append(node_global_pose_np)

        # For live visualization: render this node's cloud in "world" frame
        # using its current global guess, then apply transform_matrix for viewer
        # (this does NOT go into any global_pcd_list)
        pcd_global_for_viz = copy.deepcopy(pcd_local)
        pcd_global_for_viz.transform(node_global_pose_np)

        pcd_vis = copy.deepcopy(pcd_global_for_viz)
        pcd_vis.transform(transform_matrix)

        cam_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        cam_frame_vis.transform(node_global_pose_np)
        cam_frame_vis.transform(transform_matrix)

        geometries.append(pcd_vis)
        geometries.append(cam_frame_vis)

        last_pose_np = node_global_pose_np

    # -------------------------------------------------
    # 3. Optimize once so we start with consistent poses
    # -------------------------------------------------
    print("Running initial graph optimization...")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
    result = optimizer.optimize()

    # Update global containers with optimized values
    initial_estimates = result  # so future increments keep from here

    for node_id in range(len(poses)):
        poses[node_id] = gtsam_to_np(result.atPose3(node_id))

    print(f"Initial map created. Graph has {graph.size()} factors.")



def build_local_target_map(new_node_id,
                           camera_pose_global_guess,
                           poses,
                           pcd_database,
                           LOCAL_WINDOW_SIZE,
                           LOCAL_SEARCH_RADIUS,
                           VOXEL_SIZE):
    """
    Build a fused local map (in GLOBAL coordinates) from nearby nodes.
    Returns:
        local_map_target : o3d.geometry.PointCloud()  (downsampled)
        nearby_node_ids  : sorted list of node indices used
    """
    local_map_target = o3d.geometry.PointCloud()
    nearby_node_ids = []

    # ---- A. temporal neighbors (last few nodes) ----
    for i in range(1, LOCAL_WINDOW_SIZE + 1):
        if new_node_id - i >= 0:
            nearby_node_ids.append(new_node_id - i)

    # ---- B. spatial neighbors (nodes close in xyz) ----
    new_pose_translation = camera_pose_global_guess[0:3, 3]
    for i in range(max(0, new_node_id - LOCAL_WINDOW_SIZE)):
        old_pose_translation = poses[i][0:3, 3]
        dist = np.linalg.norm(new_pose_translation - old_pose_translation)
        if dist < LOCAL_SEARCH_RADIUS:
            nearby_node_ids.append(i)

    # Deduplicate + sort
    nearby_node_ids = sorted(list(set(nearby_node_ids)))

    # ---- C. fuse their point clouds in GLOBAL frame ----
    if not nearby_node_ids:
        print("Warning: No nearby nodes found for local map.")
    else:
        for node_id in nearby_node_ids:
            if node_id in pcd_database:
                old_cloud_local = pcd_database[node_id]  # cloud in that node's LOCAL frame
                old_cloud_global = copy.deepcopy(old_cloud_local)
                old_cloud_global.transform(poses[node_id])  # poses[node_id] is latest global est.
                local_map_target += old_cloud_global

    # ---- D. downsample so ICP doesn't die ----
    if local_map_target.has_points():
        local_map_target = local_map_target.voxel_down_sample(voxel_size=VOXEL_SIZE)

    return local_map_target, nearby_node_ids


def add_factors_for_new_node(new_node_id,
                             refined_guess_np,
                             prev_node_id,
                             poses,
                             graph,
                             initial_estimates,
                             ODOMETRY_NOISE,
                             LOOP_NOISE,
                             loop_closure_edges):
    """
    - Insert new node pose guess into initial_estimates
    - Add odometry factor (prev -> new)
    - Add loop-closure factors (ICP constraints)
    """

    # 1. Insert the node's initial pose estimate (Pose3) to GTSAM values
    initial_estimates.insert(new_node_id, np_to_gtsam(refined_guess_np))

    # 2. Odometry-like edge between prev_node_id and new_node_id
    prev_node_pose = poses[prev_node_id]  # global pose (numpy 4x4) of previous node
    relative_pose_odom = np.linalg.inv(prev_node_pose) @ refined_guess_np
    graph.add(
        gtsam.BetweenFactorPose3(
            prev_node_id,
            new_node_id,
            np_to_gtsam(relative_pose_odom),
            ODOMETRY_NOISE,
        )
    )

    # 3. Loop closure edges from ICP
    for (nid_a, nid_b, rel_pose_np) in loop_closure_edges:
        # nid_a --(rel_pose_np)--> nid_b
        graph.add(
            gtsam.BetweenFactorPose3(
                nid_a,
                nid_b,
                np_to_gtsam(rel_pose_np),
                LOOP_NOISE,
            )
        )

def mapping(preds):
 
    global graph, initial_estimates, poses, pcd_database
    global LOCAL_WINDOW_SIZE, LOCAL_SEARCH_RADIUS, VOXEL_SIZE
    global ODOMETRY_NOISE, LOOP_NOISE
    global MAX_FILTER_DISTANCE, transform_matrix
    # geometries is assumed global for visualization

    # ---------------------------------------------------------
    # 0. Take the newest prediction from this batch of preds
    # ---------------------------------------------------------
    pred = preds[len(preds) - 1]

    # ---------------------------------------------------------
    # 1. Build this node's local point cloud from pts3d_cam
    # ---------------------------------------------------------
    # pts3d_cam: points in THIS camera's local coordinates (camera = origin)
    points_local = pred["pts3d_cam"].squeeze().cpu().numpy().reshape(-1, 3)
    colors = pred["img_no_norm"].squeeze().cpu().numpy().reshape(-1, 3)

    # model's per-view pose relative to first window frame
    camera_pose_relative = pred["camera_poses"].squeeze().cpu().numpy().astype(np.float32)

    # Filter out far points to reduce junk
    distances = np.linalg.norm(points_local, axis=1)  # dist from camera (0,0,0)
    mask = distances <= MAX_FILTER_DISTANCE
    filtered_points = points_local[mask]
    filtered_colors = colors[mask]

    # Build source cloud in local camera frame
    pcd_local_source = o3d.geometry.PointCloud()
    pcd_local_source.points = o3d.utility.Vector3dVector(filtered_points)
    pcd_local_source.colors = o3d.utility.Vector3dVector(filtered_colors)

    if not pcd_local_source.has_points():
        print("Warning: No points in new cloud, skipping frame.")
        return

    # ---------------------------------------------------------
    # 2. Compute VO-style global pose guess for this node
    # ---------------------------------------------------------
    # We assume: first view in the sliding window already exists in poses[-4]
    # so we lift this camera into global coords.
    window_start_pose_index = len(poses) - 4
    global_pose_of_window_start = poses[window_start_pose_index]  # 4x4 global of anchor frame

    # camera_pose_relative is (anchor -> this new camera)
    camera_pose_global_guess = (global_pose_of_window_start @ camera_pose_relative).astype(np.float32)

    # We'll refine this guess with ICP if we can.
    refined_guess_np = camera_pose_global_guess

    # Node bookkeeping
    new_node_id = len(poses)
    prev_node_id = new_node_id - 1  # last node is our odom predecessor

    # ---------------------------------------------------------
    # 3. Build local target map for ICP using recent/spatial neighbors
    # ---------------------------------------------------------
    local_map_target, nearby_node_ids = build_local_target_map(
        new_node_id=new_node_id,
        camera_pose_global_guess=camera_pose_global_guess,
        poses=poses,
        pcd_database=pcd_database,
        LOCAL_WINDOW_SIZE=LOCAL_WINDOW_SIZE,
        LOCAL_SEARCH_RADIUS=LOCAL_SEARCH_RADIUS,
        VOXEL_SIZE=VOXEL_SIZE,
    )

    loop_closure_edges = []

    # ---------------------------------------------------------
    # 4. If we have a local target map, run ICP to align new cloud
    # ---------------------------------------------------------
    if local_map_target.has_points():
        # run_icp() will:
        #   - transform pcd_local_source by camera_pose_global_guess
        #   - then run point-to-plane ICP to compute a correction
        icp_result = run_icp(
            source_pcd=pcd_local_source,
            target_pcd_fused=local_map_target,
            initial_guess_transform=camera_pose_global_guess,
        )

        if icp_result.fitness > 0.3:
            print(f"ICP constraint accepted. Fitness={icp_result.fitness:.3f}")

            # Update the node's global pose guess using ICP correction
            corrected_global_pose = (icp_result.transformation @ camera_pose_global_guess).astype(np.float32)
            refined_guess_np = corrected_global_pose  # better than raw VO guess

            # Pick a neighbor node to anchor a loop-closure-like factor
            # (You chose "most recent one" = nearby_node_ids[-1])
            closest_node_id = nearby_node_ids[-1]

            # relative pose from closest_node -> this new node (using ICP pose)
            relative_pose_loop = np.linalg.inv(poses[closest_node_id]) @ corrected_global_pose

            loop_closure_edges.append(
                (closest_node_id, new_node_id, relative_pose_loop)
            )

    # ---------------------------------------------------------
    # 5. NOW add the node + edges to the factor graph
    #    (We delay until after ICP because we want refined_guess_np)
    # ---------------------------------------------------------
    add_factors_for_new_node(
        new_node_id=new_node_id,
        refined_guess_np=refined_guess_np,
        prev_node_id=prev_node_id,
        poses=poses,
        graph=graph,
        initial_estimates=initial_estimates,
        ODOMETRY_NOISE=ODOMETRY_NOISE,
        LOOP_NOISE=LOOP_NOISE,
        loop_closure_edges=loop_closure_edges,
    )

    # ---------------------------------------------------------
    # 6. Optimize the global graph so ALL poses (old and new) adjust
    # ---------------------------------------------------------
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
    result = optimizer.optimize()
    print(
        f"Graph optimized. Error changed from "
        f"{graph.error(initial_estimates):.3f} to {graph.error(result):.3f}"
    )

    # Update the running estimate container
    initial_estimates = result

    # Update our Python list of poses[] in numpy, for all nodes so far
    poses[:] = []  # clear and refill
    for nid in range(new_node_id + 1):
        poses.append(gtsam_to_np(result.atPose3(nid)))

    # ---------------------------------------------------------
    # 7. Store data for future ICP and visualization
    # ---------------------------------------------------------

    # 7a. save the local (untransformed) cloud for this node in the database
    #     we'll always transform it with poses[nid] LATER (final fusion)
    pcd_database[new_node_id] = pcd_local_source

    # 7b. OPTIONAL live visualization for this node:
    camera_pose_global_final = poses[new_node_id]  # optimized global pose
    pcd_global_for_viz = copy.deepcopy(pcd_local_source)
    pcd_global_for_viz.transform(camera_pose_global_final)

    # Turn into display coordinates for Open3D draw loop
    pcd_vis = copy.deepcopy(pcd_global_for_viz)
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



print("All batches processed. Fusing point clouds...")

# 1. Build final fused world cloud from final poses
global_pcd_raw = o3d.geometry.PointCloud()
for node_id, local_cloud in pcd_database.items():
    cloud_global = copy.deepcopy(local_cloud)
    cloud_global.transform(poses[node_id])  # poses[node_id] is FINAL
    global_pcd_raw += cloud_global

print(f"Total points before fusing: {len(global_pcd_raw.points)}")

fused_pcd_raw = global_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
print(f"Total points after fusing: {len(fused_pcd_raw.points)}")

# 2. Transform fused cloud into your visualization frame
fused_pcd_vis = copy.deepcopy(fused_pcd_raw)
fused_pcd_vis.transform(transform_matrix)

# 3. Build fresh camera frames from FINAL poses (not from geometries[])
camera_frames_final = []
for node_id, pose_np in enumerate(poses):
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    cam_frame.transform(pose_np)
    cam_frame.transform(transform_matrix)
    camera_frames_final.append(cam_frame)

# 4. Camera trajectory line (FINAL poses)
path_points = [p[0:3, 3] for p in poses]
path_lines = [[i, i + 1] for i in range(len(path_points) - 1)]
path_colors = [[1, 0, 0] for _ in range(len(path_lines))]  # red

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(path_points),
    lines=o3d.utility.Vector2iVector(path_lines),
)
line_set.colors = o3d.utility.Vector3dVector(path_colors)
line_set.transform(transform_matrix)

# 5. Draw only final stuff
final_geoms = [fused_pcd_vis, line_set] + camera_frames_final
o3d.visualization.draw_geometries(final_geoms)


