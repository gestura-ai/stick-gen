"use strict";

// Headless Three.js renderer for .motion files.
// Dependencies (install in your Node environment):
//   npm install three pngjs gl

const fs = require("fs");
const path = require("path");
const { PNG } = require("pngjs");
const THREE = require("three");
const createGL = require("gl");

// Environment-specific parallax layer configurations
// Each environment defines color schemes for foreground, midground, and background layers
const ENVIRONMENT_PARALLAX_CONFIGS = {
  // Earth environments
  earth_normal: [
    { depth: -12, top: 0x87ceeb, bottom: 0x4682b4 }, // sky blue
    { depth: -24, top: 0x228b22, bottom: 0x006400 }, // forest green
    { depth: -48, top: 0x2f4f4f, bottom: 0x1a1a2e }, // dark slate
  ],
  underwater: [
    { depth: -12, top: 0x00bfff, bottom: 0x006994 }, // deep blue
    { depth: -24, top: 0x004080, bottom: 0x001a33 }, // ocean depths
    { depth: -48, top: 0x000d1a, bottom: 0x000000 }, // abyss
  ],
  space: [
    { depth: -12, top: 0x0a0a20, bottom: 0x050510 }, // near space
    { depth: -24, top: 0x030308, bottom: 0x010104 }, // deep space
    { depth: -48, top: 0x000000, bottom: 0x000000 }, // void
  ],
  moon: [
    { depth: -12, top: 0x2a2a2a, bottom: 0x1a1a1a }, // lunar surface
    { depth: -24, top: 0x101010, bottom: 0x080808 }, // lunar shadows
    { depth: -48, top: 0x000000, bottom: 0x000000 }, // space backdrop
  ],
  forest: [
    { depth: -12, top: 0x90ee90, bottom: 0x228b22 }, // light green canopy
    { depth: -24, top: 0x006400, bottom: 0x003300 }, // deep forest
    { depth: -48, top: 0x1a2f1a, bottom: 0x0d1a0d }, // forest floor
  ],
  desert: [
    { depth: -12, top: 0xffd700, bottom: 0xdaa520 }, // golden sand
    { depth: -24, top: 0xcd853f, bottom: 0x8b4513 }, // dunes
    { depth: -48, top: 0x654321, bottom: 0x3d2314 }, // distant mountains
  ],
  arctic: [
    { depth: -12, top: 0xf0ffff, bottom: 0xe0f0ff }, // ice white
    { depth: -24, top: 0xb0c4de, bottom: 0x87ceeb }, // frozen blue
    { depth: -48, top: 0x4682b4, bottom: 0x2f4f4f }, // arctic depths
  ],
  volcanic: [
    { depth: -12, top: 0xff4500, bottom: 0xcc3700 }, // lava glow
    { depth: -24, top: 0x8b0000, bottom: 0x4a0000 }, // molten rock
    { depth: -48, top: 0x1a0a0a, bottom: 0x0a0505 }, // ash clouds
  ],
  urban: [
    { depth: -12, top: 0x708090, bottom: 0x4a5568 }, // city haze
    { depth: -24, top: 0x2d3748, bottom: 0x1a202c }, // buildings
    { depth: -48, top: 0x0d1117, bottom: 0x050505 }, // night sky
  ],
  cyberpunk: [
    { depth: -12, top: 0xff00ff, bottom: 0x8b008b }, // neon pink
    { depth: -24, top: 0x00ffff, bottom: 0x008b8b }, // cyan glow
    { depth: -48, top: 0x1a0a2e, bottom: 0x0a0514 }, // dark purple
  ],
  fantasy: [
    { depth: -12, top: 0xffe0f7, bottom: 0xff80c0 }, // magical pink
    { depth: -24, top: 0x80d0ff, bottom: 0x4080c0 }, // enchanted blue
    { depth: -48, top: 0x2a1a4a, bottom: 0x0a0520 }, // mystical purple
  ],
  horror: [
    { depth: -12, top: 0x4a0000, bottom: 0x2a0000 }, // blood red
    { depth: -24, top: 0x1a0a0a, bottom: 0x0a0505 }, // dark crimson
    { depth: -48, top: 0x050202, bottom: 0x000000 }, // void
  ],
  sunset: [
    { depth: -12, top: 0xff7f50, bottom: 0xff4500 }, // coral orange
    { depth: -24, top: 0xdc143c, bottom: 0x8b0000 }, // crimson
    { depth: -48, top: 0x4b0082, bottom: 0x1a0a2e }, // indigo night
  ],
  sunrise: [
    { depth: -12, top: 0xffd700, bottom: 0xffa500 }, // golden
    { depth: -24, top: 0xff8c00, bottom: 0xff6347 }, // orange
    { depth: -48, top: 0x4169e1, bottom: 0x191970 }, // royal blue
  ],
  storm: [
    { depth: -12, top: 0x4a5568, bottom: 0x2d3748 }, // storm gray
    { depth: -24, top: 0x1a202c, bottom: 0x0d1117 }, // dark clouds
    { depth: -48, top: 0x050505, bottom: 0x000000 }, // black sky
  ],
  cave: [
    { depth: -12, top: 0x3d3d3d, bottom: 0x2a2a2a }, // cave entrance
    { depth: -24, top: 0x1a1a1a, bottom: 0x0d0d0d }, // deep cave
    { depth: -48, top: 0x050505, bottom: 0x000000 }, // darkness
  ],
  beach: [
    { depth: -12, top: 0x87ceeb, bottom: 0x00bfff }, // sky blue
    { depth: -24, top: 0xf4a460, bottom: 0xdeb887 }, // sandy
    { depth: -48, top: 0x006994, bottom: 0x004080 }, // ocean
  ],
  mountain: [
    { depth: -12, top: 0x87ceeb, bottom: 0x4682b4 }, // sky
    { depth: -24, top: 0x696969, bottom: 0x4a4a4a }, // rock gray
    { depth: -48, top: 0x2f4f4f, bottom: 0x1a1a2e }, // distant peaks
  ],
  swamp: [
    { depth: -12, top: 0x6b8e23, bottom: 0x556b2f }, // olive green
    { depth: -24, top: 0x3d5c3d, bottom: 0x2f4f2f }, // murky green
    { depth: -48, top: 0x1a2f1a, bottom: 0x0d1a0d }, // swamp depths
  ],
  jungle: [
    { depth: -12, top: 0x32cd32, bottom: 0x228b22 }, // lime green
    { depth: -24, top: 0x006400, bottom: 0x004d00 }, // deep jungle
    { depth: -48, top: 0x002600, bottom: 0x001300 }, // jungle floor
  ],
  // Default fallback
  default: [
    { depth: -12, top: 0xffe0f7, bottom: 0xff80c0 }, // foreground candy sky
    { depth: -24, top: 0x80d0ff, bottom: 0x2040a0 }, // midground city
    { depth: -48, top: 0x101020, bottom: 0x050509 }, // deep space
  ],
};

// Environment-specific camera behavior modifiers
const ENVIRONMENT_CAMERA_MODIFIERS = {
  underwater: { radiusScale: 0.8, heightScale: 0.7, fovOffset: 5, movementScale: 0.5 },
  space: { radiusScale: 1.5, heightScale: 1.2, fovOffset: -10, movementScale: 0.3 },
  moon: { radiusScale: 1.3, heightScale: 1.0, fovOffset: -5, movementScale: 0.4 },
  cave: { radiusScale: 0.6, heightScale: 0.5, fovOffset: 10, movementScale: 0.6 },
  forest: { radiusScale: 0.9, heightScale: 0.8, fovOffset: 5, movementScale: 0.8 },
  urban: { radiusScale: 1.0, heightScale: 1.2, fovOffset: 0, movementScale: 1.0 },
  cyberpunk: { radiusScale: 1.0, heightScale: 1.1, fovOffset: 5, movementScale: 1.2 },
  storm: { radiusScale: 1.1, heightScale: 0.9, fovOffset: 0, movementScale: 1.5 },
  default: { radiusScale: 1.0, heightScale: 1.0, fovOffset: 0, movementScale: 1.0 },
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
  renderer.setClearColor(0x000000, 1.0);
  renderer.outputEncoding = THREE.sRGBEncoding;
  return { renderer, gl };
}

function createSkeleton() {
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array(10 * 3); // 5 segments * 2 endpoints
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color: 0xffffff,
    linewidth: 3,
    transparent: true,
    opacity: 0.95,
  });
  const lines = new THREE.LineSegments(geom, material);
  return { lines, positions, geom };
}

function createParallaxLayers(environmentType) {
  const group = new THREE.Group();
  // Select environment-specific parallax configuration
  const configs =
    ENVIRONMENT_PARALLAX_CONFIGS[environmentType] || ENVIRONMENT_PARALLAX_CONFIGS.default;
  configs.forEach((cfg) => {
    const geo = new THREE.PlaneGeometry(40, 40, 1, 1);
    const colors = [];
    const top = new THREE.Color(cfg.top);
    const bottom = new THREE.Color(cfg.bottom);
    for (let i = 0; i < 4; i += 1) {
      const c = i < 2 ? top : bottom;
      colors.push(c.r, c.g, c.b);
    }
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const mat = new THREE.MeshBasicMaterial({ vertexColors: true, depthWrite: false });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(0, 0, cfg.depth);
    group.add(mesh);
  });
  return group;
}

function getCameraModifiers(environmentType) {
  return ENVIRONMENT_CAMERA_MODIFIERS[environmentType] || ENVIRONMENT_CAMERA_MODIFIERS.default;
}

function createProps() {
  const group = new THREE.Group();
  const ballGeom = new THREE.SphereGeometry(0.18, 16, 16);
  const ballMat = new THREE.MeshBasicMaterial({ color: 0xffff80 });
  const platformGeom = new THREE.BoxGeometry(4, 0.3, 0.5);
  const platformMat = new THREE.MeshBasicMaterial({ color: 0x303040 });
  const trunkGeom = new THREE.BoxGeometry(0.25, 1.3, 0.25);
  const leafGeom = new THREE.SphereGeometry(0.8, 12, 12);
  const trunkMat = new THREE.MeshBasicMaterial({ color: 0x8b4513 });
  const leafMat = new THREE.MeshBasicMaterial({ color: 0x2e8b57 });
  return { group, ballGeom, ballMat, platformGeom, platformMat, trunkGeom, leafGeom, trunkMat, leafMat };
}

function computeJoints(frame) {
  return {
    neck: { x: frame[0], y: frame[1] },
    hip: { x: frame[2], y: frame[3] },
    leftHand: { x: frame[14], y: frame[15] },
    rightHand: { x: frame[18], y: frame[19] },
  };
}

function updateSkeleton(skel, frame) {
  const zDepths = [0.0, -0.3, 0.3, -0.5, 0.5];
  const pos = skel.positions;
  for (let seg = 0; seg < 5; seg += 1) {
    const base = seg * 4;
    const vBase = seg * 6;
    const z = zDepths[seg] || 0;
    const x1 = frame[base];
    const y1 = frame[base + 1];
    const x2 = frame[base + 2];
    const y2 = frame[base + 3];
    pos[vBase] = x1;
    pos[vBase + 1] = y1;
    pos[vBase + 2] = z;
    pos[vBase + 3] = x2;
    pos[vBase + 4] = y2;
    pos[vBase + 5] = z;
  }
  skel.geom.attributes.position.needsUpdate = true;
}

function updateProps(props, frame) {
  const joints = computeJoints(frame);
  while (props.group.children.length) props.group.remove(props.group.children[0]);
  const r = Math.random();
  if (r < 0.33) {
    const ball = new THREE.Mesh(props.ballGeom, props.ballMat);
    ball.position.set(joints.rightHand.x, joints.rightHand.y, 0.25);
    props.group.add(ball);
  } else if (r < 0.66) {
    const plat = new THREE.Mesh(props.platformGeom, props.platformMat);
    plat.position.set(joints.hip.x, joints.hip.y - 1.3, 0.0);
    props.group.add(plat);
  } else {
    const trunk = new THREE.Mesh(props.trunkGeom, props.trunkMat);
    const leaves = new THREE.Mesh(props.leafGeom, props.leafMat);
    trunk.position.set(joints.hip.x + 3.0, joints.hip.y - 0.3, -0.2);
    leaves.position.set(joints.hip.x + 3.0, joints.hip.y + 0.6, -0.2);
    props.group.add(trunk);
    props.group.add(leaves);
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
      "Usage: node threejs_parallax_renderer.js --input motion.motion --output-dir out --views 100 [--frames-per-view 8] [--environment-type underwater]"
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
  // Environment type for environment-aware parallax and camera behavior
  const environmentType = args.environment_type || "default";

  const { frames, meta } = loadMotion(inputPath);
  if (!frames.length) {
    console.error("No frames in motion file.");
    process.exit(1);
  }

  // Get environment-specific camera modifiers
  const camMod = getCameraModifiers(environmentType);

  const { renderer, gl } = createRenderer(width, height);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(0, 2, 10);
  camera.lookAt(0, 1, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 10, 10);
  scene.add(ambient);
  scene.add(dir);

  // Create environment-aware parallax layers
  const parallax = createParallaxLayers(environmentType);
  const skel = createSkeleton();
  const props = createProps();
  scene.add(parallax);
  scene.add(skel.lines);
  scene.add(props.group);

  const metaOut = {
    sample_id: sampleId,
    actor_id: actorId,
    environment_type: environmentType,
    views,
    frames_per_view: framesPerView,
    motion_total_frames: frames.length,
    fps: meta.fps || 25,
    frames: [],
  };

  for (let i = 0; i < views; i += 1) {
    const baseFrame = Math.floor(Math.random() * frames.length);

    // Apply environment-specific camera modifiers
    const baseAngle = Math.random() * Math.PI * 2;
    const angleDelta = (Math.random() * 0.4 - 0.2) * camMod.movementScale;
    const baseRadius = (8 + Math.random() * 4) * camMod.radiusScale;
    const radiusDelta = (Math.random() * 0.4 - 0.2) * camMod.movementScale;
    const baseHeight = (1 + Math.random() * 2.5) * camMod.heightScale;
    const heightDelta = (Math.random() * 0.3 - 0.15) * camMod.movementScale;
    const baseFov = 35 + Math.random() * 20 + camMod.fovOffset;
    const fovDelta = (Math.random() * 6 - 3) * camMod.movementScale;

    for (let step = 0; step < framesPerView; step += 1) {
      const motionIdx = Math.min(baseFrame + step, frames.length - 1);
      const frame = frames[motionIdx];
      updateSkeleton(skel, frame);
      updateProps(props, frame);

      const tNorm = framesPerView > 1 ? step / (framesPerView - 1) : 0.0;
      const angle = baseAngle + angleDelta * tNorm;
      const radius = baseRadius + radiusDelta * tNorm;
      const heightCam = baseHeight + heightDelta * tNorm;
      const fov = baseFov + fovDelta * tNorm;

      camera.fov = fov;
      camera.updateProjectionMatrix();
      camera.position.set(Math.cos(angle) * radius, heightCam, Math.sin(angle) * radius);
      camera.lookAt(0, 1, 0);

      renderer.render(scene, camera);
      const frameName =
        framesPerView > 1
          ? `view_${String(i).padStart(5, "0")}_f${String(step).padStart(3, "0")}.png`
          : `view_${String(i).padStart(5, "0")}.png`;
      const outPath = path.join(outputDir, frameName);
      saveFrame(gl, width, height, outPath);

      metaOut.frames.push({
        file: frameName,
        view_index: i, // Deprecated alias for view_id (kept for backward compatibility)
        view_id: i,
        step_index: step,
        motion_frame_index: motionIdx,
        environment_type: environmentType,
        camera: {
          position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
          target: { x: 0, y: 1, z: 0 },
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

