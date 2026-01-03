"use strict";

// Headless Three.js renderer for .motion files.
// Dependencies (install in your Node environment):
//   npm install three pngjs gl

const fs = require("fs");
const path = require("path");
const { PNG } = require("pngjs");
const THREE = require("three");
const createGL = require("gl");

// =============================================================================
// JOINT-BASED Z-DEPTHS
// =============================================================================
// Each joint has a consistent z-depth so that connected segments meet in 3D.

const JOINT_Z_DEPTHS = {
  // Axial chain (centered)
  head_center: 0.0,
  neck: 0.0,
  chest: 0.0,
  pelvis_center: 0.0,
  // Left side (negative z = towards camera left)
  l_shoulder: -0.12,
  l_elbow: -0.18,
  l_wrist: -0.22,
  l_hip: -0.08,
  l_knee: -0.12,
  l_ankle: -0.14,
  // Right side (positive z = towards camera right)
  r_shoulder: 0.12,
  r_elbow: 0.18,
  r_wrist: 0.22,
  r_hip: 0.08,
  r_knee: 0.12,
  r_ankle: 0.14,
};

// V3 segment definitions: [segmentIndex, startJoint, endJoint]
// These define the TOPOLOGY. We draw lines between these SHARED joints.
const V3_SEGMENT_TOPOLOGY = [
  { start: "neck", end: "head_center" },       // head
  { start: "neck", end: "chest" },             // upper_torso
  { start: "chest", end: "pelvis_center" },    // lower_torso
  { start: "l_shoulder", end: "l_elbow" },     // l_upper_arm
  { start: "l_elbow", end: "l_wrist" },        // l_forearm
  { start: "r_shoulder", end: "r_elbow" },     // r_upper_arm
  { start: "r_elbow", end: "r_wrist" },        // r_forearm
  { start: "l_hip", end: "l_knee" },           // l_thigh
  { start: "l_knee", end: "l_ankle" },         // l_shin
  { start: "r_hip", end: "r_knee" },           // r_thigh
  { start: "r_knee", end: "r_ankle" },         // r_shin
  { start: "l_hip", end: "r_hip" },            // pelvis_width
];

// Visual-only connections to fix "floating limbs" look
// These are rendered but not part of the underlying data format.
const VISUAL_CONNECTIONS = [
  { start: "neck", end: "l_shoulder" },       // Left clavicle
  { start: "neck", end: "r_shoulder" },       // Right clavicle
  { start: "pelvis_center", end: "l_hip" },   // Left pelvic strut
  { start: "pelvis_center", end: "r_hip" },   // Right pelvic strut
];

// Combine them into a single render topology
const FULL_RENDER_TOPOLOGY = [...V3_SEGMENT_TOPOLOGY, ...VISUAL_CONNECTIONS];


// Environment-specific parallax layer configurations (kept minimal for --minimal mode)
const ENVIRONMENT_PARALLAX_CONFIGS = {
  default: [
    { depth: -15, color: 0x1a1a2e },
  ],
  minimal: [
    { depth: -20, color: 0x0a0a0a },
  ],
};

function parseArgs(argv) {
  const opts = {};
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2).replace(/-/g, "_");
      const next = argv[i + 1];
      if (!next || next.startsWith("--")) {
        opts[key] = true;
      } else {
        opts[key] = next;
        i += 1;
      }
    }
  }
  return opts;
}

function loadMotion(motionPath) {
  const raw = JSON.parse(fs.readFileSync(motionPath, "utf8"));
  const meta = raw.meta || {};
  const skel = raw.skeleton || {};
  const stride = skel.input_dim || 20;
  const flat = raw.motion || [];
  const totalFrames = meta.total_frames || Math.floor(flat.length / stride);
  const frames = new Array(totalFrames);
  for (let i = 0; i < totalFrames; i += 1) {
    frames[i] = flat.slice(i * stride, (i + 1) * stride);
  }
  return { meta, skel, frames, actions: raw.actions || [] };
}

function getSkeletonInfo(skel) {
  const inputDim = skel.input_dim || 20;
  const segments = Array.isArray(skel.segments) ? skel.segments : [];
  let numSegments = segments.length;
  if (!numSegments && inputDim > 0) {
    numSegments = Math.floor(inputDim / 4);
  }
  const type = skel.type || "stick_figure_5_segment";
  const isV3 =
    inputDim === 48 &&
    numSegments === 12 &&
    typeof type === "string" &&
    type.indexOf("12_segment_v3") !== -1;

  return {
    inputDim,
    segments,
    numSegments,
    type,
    isV3,
  };
}

// Helper: Apply strict anthropometric clamping relative to PARENT JOINTS
function applyAnthropometricClamping(joints) {
  const neck = joints.neck;
  const pelvis = joints.pelvis_center;

  if (!neck || !pelvis) return joints;

  // Calculate Torso Length (Spine)
  const dx = neck.x - pelvis.x;
  const dy = neck.y - pelvis.y;
  let torsoLen = Math.sqrt(dx * dx + dy * dy);
  if (torsoLen < 0.01) torsoLen = 0.3; // Safety floor

  // RATIOS (Relative to Parent):
  // Shoulders (vs Neck): Max 0.6x Torso (Clavicle width)
  // Arms (Shoulder->Wrist): Max 1.3x Torso (Human arm is longer than torso)
  // Legs (Hip->Ankle): Max 1.5x Torso
  const MAX_SHOULDER_RATIO = 0.6;
  const MAX_ARM_RATIO = 1.3;
  const MAX_LEG_RATIO = 1.5;

  const clamp = (joint, ratio, origin) => {
    if (!joint || !origin) return joint;
    const maxDist = torsoLen * ratio;
    const jdx = joint.x - origin.x;
    const jdy = joint.y - origin.y;
    const dist = Math.sqrt(jdx * jdx + jdy * jdy);
    if (dist > maxDist) {
      const scale = maxDist / dist;
      return { x: origin.x + jdx * scale, y: origin.y + jdy * scale };
    }
    return joint;
  };

  // Clamp Shoulders relative to Neck
  joints.l_shoulder = clamp(joints.l_shoulder, MAX_SHOULDER_RATIO, neck);
  joints.r_shoulder = clamp(joints.r_shoulder, MAX_SHOULDER_RATIO, neck);

  // Clamp Arms relative to Shoulders (Crucial for symmetry)
  joints.l_wrist = clamp(joints.l_wrist, MAX_ARM_RATIO, joints.l_shoulder);
  joints.r_wrist = clamp(joints.r_wrist, MAX_ARM_RATIO, joints.r_shoulder);
  joints.l_elbow = clamp(joints.l_elbow, MAX_ARM_RATIO * 0.6, joints.l_shoulder); // Elbow halfway
  joints.r_elbow = clamp(joints.r_elbow, MAX_ARM_RATIO * 0.6, joints.r_shoulder);

  // Clamp Legs relative to Hips
  joints.l_ankle = clamp(joints.l_ankle, MAX_LEG_RATIO, joints.l_hip);
  joints.r_ankle = clamp(joints.r_ankle, MAX_LEG_RATIO, joints.r_hip);
  joints.l_knee = clamp(joints.l_knee, MAX_LEG_RATIO * 0.6, joints.l_hip);
  joints.r_knee = clamp(joints.r_knee, MAX_LEG_RATIO * 0.6, joints.r_hip);

  // Head Rectification:
  // After alignment, Neck is strictly above Pelvis (Frame Up).
  // Head must be above Neck. If data puts Head below Neck (e.g. in Chest), force it up.
  const head = joints.head_center;
  if (head) {
    // We expect Head.y > Neck.y
    // And ideally vertical alignment since it's a stick figure T-pose-ish
    const minHeadHeight = torsoLen * 0.2; // Head ~ 20% of torso size

    // Project Head onto Spine Axis (which is now Y-axis after alignment)
    // If Head is too low, snap it up.
    if (head.y < neck.y + minHeadHeight) {
      // Force Head Position
      // Keep original X offset if reasonable, else center it
      let hx = head.x;
      if (Math.abs(hx - neck.x) > torsoLen * 0.3) hx = neck.x; // Center if straying too far

      joints.head_center = {
        x: hx,
        y: neck.y + minHeadHeight
      };
    }
  }

  return joints;
}

// Helper: Align skeleton so spine points strictly UP (0, 1)
// This resolves sideways, upside-down, and arbitrary rotation issues.
function alignSkeletonToUpright(joints) {
  const neck = joints.neck;
  const pelvis = joints.pelvis_center;

  if (!neck || !pelvis) return joints;

  // Pivot around Pelvis
  // Current Spine Vector
  const dx = neck.x - pelvis.x;
  const dy = neck.y - pelvis.y;
  const currentAngle = Math.atan2(dy, dx);

  // Target Angle: +PI/2 (Up in 2D Euclidean, which maps to Up in Y-Up 3D)
  const targetAngle = Math.PI / 2;
  const rotation = targetAngle - currentAngle;

  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);

  // Rotate point p around origin o
  const rotatePoint = (p, o) => {
    const tx = p.x - o.x;
    const ty = p.y - o.y;
    return {
      x: o.x + (tx * cos - ty * sin),
      y: o.y + (tx * sin + ty * cos)
    };
  };

  // Apply to all joints
  Object.keys(joints).forEach(key => {
    if (key === 'pelvis_center') return; // Pivot point stays put
    joints[key] = rotatePoint(joints[key], pelvis);
  });

  return joints;
}

/**
 * Data Quality Fix: Separate degenerate paired joints.
 * When L/R joints are nearly identical, force separation using body axis.
 */
function separateDegenerateJoints(joints, rx, ry) {
  const DEGENERATE_THRESHOLD = 0.05;
  const FORCED_OFFSET = 0.15;

  // Check knee degeneracy
  const l_knee = joints.l_knee;
  const r_knee = joints.r_knee;
  if (l_knee && r_knee) {
    const dx = l_knee.x - r_knee.x;
    const dy = l_knee.y - r_knee.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < DEGENERATE_THRESHOLD) {
      // Both knees are at same point - treat as center, force separation
      const centerX = (l_knee.x + r_knee.x) / 2;
      const centerY = (l_knee.y + r_knee.y) / 2;
      joints.l_knee = { x: centerX - rx * FORCED_OFFSET, y: centerY - ry * FORCED_OFFSET };
      joints.r_knee = { x: centerX + rx * FORCED_OFFSET, y: centerY + ry * FORCED_OFFSET };
    }
  }

  return joints;
}

/**
 * Data Quality Fix: Clamp outlier joints to reasonable bounds.
 * Prevents chaotic appearance from extreme joint positions.
 */
function clampOutlierJoints(joints) {
  const MAX_LIMB_DISTANCE = 1.2;
  const pelvis = joints.pelvis_center;

  if (!pelvis) return joints;

  // List of limb extremities to check
  const limbJoints = [
    'l_wrist', 'r_wrist', 'l_ankle', 'r_ankle',
    'l_elbow', 'r_elbow', 'l_knee', 'r_knee',
    'l_shoulder', 'r_shoulder', 'l_hip', 'r_hip'
  ];

  limbJoints.forEach(name => {
    const joint = joints[name];
    if (!joint) return;

    const dx = joint.x - pelvis.x;
    const dy = joint.y - pelvis.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist > MAX_LIMB_DISTANCE) {
      // Clamp to max distance, preserve direction
      const scale = MAX_LIMB_DISTANCE / dist;
      joints[name] = {
        x: pelvis.x + dx * scale,
        y: pelvis.y + dy * scale
      };
    }
  });

  return joints;
}

/**
 * Extract canonical joints from a v3 48D frame.
 */
function extractJointsFromV3Frame(frame) {
  const joints = {
    neck: { x: frame[0], y: frame[1] },
    head_center: { x: frame[2], y: frame[3] },
    chest: { x: frame[6], y: frame[7] },
    pelvis_center: { x: frame[10], y: frame[11] },
    l_shoulder: { x: frame[12], y: frame[13] },
    l_elbow: { x: frame[14], y: frame[15] },
    l_wrist: { x: frame[18], y: frame[19] },
    r_shoulder: { x: frame[20], y: frame[21] },
    r_elbow: { x: frame[22], y: frame[23] },
    r_wrist: { x: frame[26], y: frame[27] },
    l_hip: { x: frame[44], y: frame[45] },
    r_hip: { x: frame[46], y: frame[47] },
    l_knee: { x: frame[30], y: frame[31] },
    l_ankle: { x: frame[34], y: frame[35] },
    r_knee: { x: frame[38], y: frame[39] },
    r_ankle: { x: frame[42], y: frame[43] },
  };

  // Apply data quality fixes
  const neck = joints.neck;
  const pelvis = joints.pelvis_center;
  let dx = neck.x - pelvis.x;
  let dy = neck.y - pelvis.y;
  let len = Math.sqrt(dx * dx + dy * dy);
  let rx = 1, ry = 0;
  if (len > 0.0001) {
    rx = dy / len;
    ry = -dx / len;
  }

  separateDegenerateJoints(joints, rx, ry);

  // 1. Align to Upright (Fixes rotation/upside-down)
  alignSkeletonToUpright(joints);

  // 2. Strict Anthropometric Clamping (Fixes exploded limbs)
  applyAnthropometricClamping(joints);

  return joints;
}

/**
 * Extract joints from legacy 20D frame (10 Keypoints).
 * "Inflates" the skeleton to full V3 joints by inferring Shoulders/Hips,
 * but uses EXPLICIT Knees/Elbows from the data.
 */
function extractJointsFromLegacyFrame(frame) {
  // Legacy 20D Mapping Analysis:
  // 0,1: Neck
  // 2,3: Pelvis 
  // 4,5: L_Knee (Explicit in data)
  // 6,7: L_Ankle
  // 8,9: R_Knee (Explicit in data)
  // 10,11: R_Ankle
  // 12,13: L_Arm_Start (Collapses to Neck) -> IGNORE
  // 14,15: L_Wrist
  // 16,17: R_Arm_Start (Collapses to Neck) -> IGNORE
  // 18,19: R_Wrist

  // NOTE: Previous "Inversion Detection" removed.
  // alignSkeletonToUpright() now handles all rotations (upside-down, sideways, etc).

  // Read raw points (No scaleY)
  const getPt = (idx) => ({ x: frame[idx], y: frame[idx + 1] });

  const neck = getPt(0);
  const pelvis = getPt(2);
  let l_knee_explicit = getPt(4);
  let l_ankle = getPt(6);
  let r_knee_explicit = getPt(8);
  let r_ankle = getPt(10);
  let l_wrist = getPt(14);
  let r_wrist = getPt(18);

  // NOTE: Early clamping removed. Final parent-relative clamping handles this better.

  // Estimate Spine Vector first to get body scale
  let dx = neck.x - pelvis.x;
  const dy = neck.y - pelvis.y;
  let torsoLen = Math.sqrt(dx * dx + dy * dy);

  // Safety floor for torso length to prevent divide-by-zero
  if (torsoLen < 0.01) torsoLen = 0.3;

  let len = torsoLen; // Use torsoLen as the base length for axis calculation

  let ux = 0, uy = 1;
  let rx = 1, ry = 0;

  if (len > 0.0001) {
    ux = dx / len;
    uy = dy / len;
    rx = uy;
    ry = -ux;
  }

  // Inflate Shoulders and Hips
  const shoulderOffset = 0.12;
  const pelvisOffset = 0.09;

  const l_shoulder = {
    x: neck.x - rx * shoulderOffset,
    y: neck.y - ry * shoulderOffset
  };
  const r_shoulder = {
    x: neck.x + rx * shoulderOffset,
    y: neck.y + ry * shoulderOffset
  };

  const l_hip = {
    x: pelvis.x - rx * pelvisOffset,
    y: pelvis.y - ry * pelvisOffset
  };
  const r_hip = {
    x: pelvis.x + rx * pelvisOffset,
    y: pelvis.y + ry * pelvisOffset
  };

  // HYBRID LOGIC:
  // Arms: Infer Elbows from clamped wrists
  const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  const l_elbow = mid(l_shoulder, l_wrist);
  const r_elbow = mid(r_shoulder, r_wrist);

  // Legs: Use Explicit Knees (already clamped above)
  const l_knee = l_knee_explicit;
  const r_knee = r_knee_explicit;

  // Infer Chest and HeadCenter
  const chest = mid(neck, pelvis);
  const head_center = {
    x: neck.x + ux * 0.15,
    y: neck.y + uy * 0.15
  };

  const joints = {
    neck, head_center, chest, pelvis_center: pelvis,
    l_shoulder, l_elbow, l_wrist,
    r_shoulder, r_elbow, r_wrist,
    l_hip, l_knee, l_ankle,
    r_hip, r_knee, r_ankle
  };

  // Apply data quality fixes (rx, ry already computed above)
  separateDegenerateJoints(joints, rx, ry);

  // 1. Align to Upright
  alignSkeletonToUpright(joints);

  // 2. Clamping
  applyAnthropometricClamping(joints);

  return joints;
}

function createRenderer(width, height) {
  const gl = createGL(width, height, { preserveDrawingBuffer: true });
  const canvas = {
    width,
    height,
    style: {},
    addEventListener() { },
    removeEventListener() { },
    getContext() {
      return gl;
    },
  };
  const renderer = new THREE.WebGLRenderer({ canvas, context: gl, antialias: true });
  renderer.setSize(width, height, false);
  renderer.setClearColor(0x0a0a0a, 1.0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  return { renderer, gl };
}

/**
 * Create skeleton geometry.
 */
function createSkeleton() {
  const segCount = FULL_RENDER_TOPOLOGY.length;
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array(segCount * 2 * 3); // segments * 2 endpoints * 3 coords
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const material = new THREE.LineBasicMaterial({
    color: 0xffffff,
    linewidth: 2,
    transparent: false,
  });
  const lines = new THREE.LineSegments(geom, material);
  return { lines, positions, geom, segmentCount: segCount };
}

// NOTE: createJointSpheres has been removed as requested.

function createHead() {
  const group = new THREE.Group();
  const sphereMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  // Add a larger sphere for the head
  const headGeom = new THREE.SphereGeometry(0.12, 12, 12);
  const headMesh = new THREE.Mesh(headGeom, sphereMat);
  headMesh.name = "head_visual";
  group.add(headMesh);
  return { group, mesh: headMesh };
}

function createBackground(minimal) {
  const group = new THREE.Group();
  const configs = minimal
    ? ENVIRONMENT_PARALLAX_CONFIGS.minimal
    : ENVIRONMENT_PARALLAX_CONFIGS.default;

  configs.forEach((cfg) => {
    const geo = new THREE.PlaneGeometry(100, 100, 1, 1);
    const mat = new THREE.MeshBasicMaterial({
      color: cfg.color,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(0, 0, cfg.depth);
    group.add(mesh);
  });

  return group;
}

/**
 * Update skeleton with joint-based 3D positioning.
 */
function updateSkeleton(skel, frame, skelInfo, headObj) {
  const SCALE = 3.0;
  const pos = skel.positions;

  // 1. Extract raw 2D joints (x,y)
  const joints2D = skelInfo.isV3
    ? extractJointsFromV3Frame(frame)
    : extractJointsFromLegacyFrame(frame);

  // 2. Compute 3D positions for ALL joints first.
  const joints3D = {};
  Object.keys(joints2D).forEach(name => {
    const j2d = joints2D[name];
    const z = JOINT_Z_DEPTHS[name] || 0.0;
    joints3D[name] = {
      x: j2d.x * SCALE,
      y: j2d.y * SCALE,
      z: z
    };
  });

  // 3. Reconstruct lines using FULL topology.
  FULL_RENDER_TOPOLOGY.forEach((segment, i) => {
    const startJ = joints3D[segment.start];
    const endJ = joints3D[segment.end];

    if (!startJ || !endJ) return;

    // Line segment start
    const vBase = i * 6;
    pos[vBase + 0] = startJ.x;
    pos[vBase + 1] = startJ.y;
    pos[vBase + 2] = startJ.z;

    // Line segment end
    pos[vBase + 3] = endJ.x;
    pos[vBase + 4] = endJ.y;
    pos[vBase + 5] = endJ.z;
  });

  skel.geom.attributes.position.needsUpdate = true;

  // 4. Update head position
  if (headObj && joints3D.head_center) {
    headObj.mesh.position.set(
      joints3D.head_center.x,
      joints3D.head_center.y,
      joints3D.head_center.z
    );
  }
}

function saveFrame(gl, width, height, outPath) {
  const pixels = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const flipped = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const src = (height - 1 - y) * width * 4;
    const dst = y * width * 4;
    flipped.set(pixels.subarray(src, src + width * 4), dst);
  }
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const png = new PNG({ width, height });
  png.data = Buffer.from(flipped);
  png.pack().pipe(fs.createWriteStream(outPath));
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = args.input;
  if (!inputPath) {
    console.error(
      "Usage: node threejs_parallax_renderer.js --input motion.motion --output-dir out --views 100 [--minimal]"
    );
    process.exit(1);
  }

  const outputDir = args.output_dir || "parallax_frames";
  const views = parseInt(args.views || "100", 10);
  const width = parseInt(args.width || "512", 10);
  const height = parseInt(args.height || "512", 10);
  const framesPerView = parseInt(args.frames_per_view || "1", 10);
  const sampleId = args.sample_id || null;
  const actorId = args.actor_id || null;
  const metadataPath = args.metadata || path.join(outputDir, "metadata.json");
  const minimal = args.minimal === true || args.minimal === "true";

  const { frames, meta, skel } = loadMotion(inputPath);
  if (!frames.length) {
    console.error("No frames in motion file.");
    process.exit(1);
  }

  const skelInfo = getSkeletonInfo(skel);

  const { renderer, gl } = createRenderer(width, height);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(0, 2, 8);
  camera.lookAt(0, 1, 0);

  // Simple ambient lighting
  const ambient = new THREE.AmbientLight(0xffffff, 1.0);
  scene.add(ambient);

  // Create scene elements
  const background = createBackground(minimal);
  const skelObj = createSkeleton();
  const headObj = createHead();

  scene.add(background);
  scene.add(skelObj.lines);
  scene.add(headObj.group);

  const metaOut = {
    sample_id: sampleId,
    actor_id: actorId,
    minimal_mode: minimal,
    views,
    frames_per_view: framesPerView,
    motion_total_frames: frames.length,
    fps: meta.fps || 25,
    skeleton_type: skelInfo.isV3 ? "v3_12_segment" : "legacy_5_segment",
    frames: [],
  };

  for (let i = 0; i < views; i += 1) {
    const baseFrame = Math.floor(Math.random() * frames.length);

    // Camera orbit parameters
    const baseAngle = Math.random() * Math.PI * 2;
    const angleDelta = (Math.random() * 0.3 - 0.15);
    const baseRadius = 6 + Math.random() * 3;
    const radiusDelta = (Math.random() * 0.3 - 0.15);
    const baseHeight = 1.5 + Math.random() * 2;
    const heightDelta = (Math.random() * 0.2 - 0.1);
    const baseFov = 40 + Math.random() * 15;
    const fovDelta = (Math.random() * 4 - 2);

    for (let step = 0; step < framesPerView; step += 1) {
      const motionIdx = Math.min(baseFrame + step, frames.length - 1);
      const frame = frames[motionIdx];
      updateSkeleton(skelObj, frame, skelInfo, headObj);

      const tNorm = framesPerView > 1 ? step / (framesPerView - 1) : 0.0;
      const angle = baseAngle + angleDelta * tNorm;
      const radius = baseRadius + radiusDelta * tNorm;
      const heightCam = baseHeight + heightDelta * tNorm;
      const fov = baseFov + fovDelta * tNorm;

      camera.fov = fov;
      camera.updateProjectionMatrix();
      camera.position.set(Math.cos(angle) * radius, heightCam, Math.sin(angle) * radius);
      camera.lookAt(0, 1.5, 0);

      renderer.render(scene, camera);
      const frameName =
        framesPerView > 1
          ? `view_${String(i).padStart(5, "0")}_f${String(step).padStart(3, "0")}.png`
          : `view_${String(i).padStart(5, "0")}.png`;
      const outPath = path.join(outputDir, frameName);
      saveFrame(gl, width, height, outPath);

      metaOut.frames.push({
        file: frameName,
        view_id: i,
        step_index: step,
        motion_frame_index: motionIdx,
        camera: {
          position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
          target: { x: 0, y: 1.5, z: 0 },
          fov: camera.fov,
        },
      });
    }
  }

  fs.mkdirSync(path.dirname(metadataPath), { recursive: true });
  fs.writeFileSync(metadataPath, JSON.stringify(metaOut, null, 2), "utf8");
}

if (require.main === module) {
  main();
}
