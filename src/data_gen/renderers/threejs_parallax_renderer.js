"use strict";

// Headless Three.js renderer for .motion files.
// Dependencies (install in your Node environment):
//   npm install three pngjs gl

const fs = require("fs");
const path = require("path");
const { PNG } = require("pngjs");
const THREE = require("three");
const createGL = require("gl");

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
    addEventListener() {},
    removeEventListener() {},
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

function createParallaxLayers() {
  const group = new THREE.Group();
  const configs = [
    { depth: -12, top: 0xffe0f7, bottom: 0xff80c0 }, // foreground candy sky
    { depth: -24, top: 0x80d0ff, bottom: 0x2040a0 }, // midground city
    { depth: -48, top: 0x101020, bottom: 0x050509 }, // deep space
  ];
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
    console.error("Usage: node threejs_parallax_renderer.js --input motion.motion --output-dir out --views 100 [--frames-per-view 8]");
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

  const { frames, meta } = loadMotion(inputPath);
  if (!frames.length) {
    console.error("No frames in motion file.");
    process.exit(1);
  }

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

  const parallax = createParallaxLayers();
  const skel = createSkeleton();
  const props = createProps();
  scene.add(parallax);
  scene.add(skel.lines);
  scene.add(props.group);

  const metaOut = {
    sample_id: sampleId,
    actor_id: actorId,
    views,
    frames_per_view: framesPerView,
    motion_total_frames: frames.length,
    fps: meta.fps || 25,
    frames: [],
  };

  for (let i = 0; i < views; i += 1) {
    const baseFrame = Math.floor(Math.random() * frames.length);

    const baseAngle = Math.random() * Math.PI * 2;
    const angleDelta = Math.random() * 0.4 - 0.2;
    const baseRadius = 8 + Math.random() * 4;
    const radiusDelta = Math.random() * 0.4 - 0.2;
    const baseHeight = 1 + Math.random() * 2.5;
    const heightDelta = Math.random() * 0.3 - 0.15;
    const baseFov = 35 + Math.random() * 20;
    const fovDelta = Math.random() * 6 - 3;

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
      const frameName = framesPerView > 1
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

