import jax.numpy as jnp
from brax import envs


class StickFigureEnv(envs.Env):
    """
    A 2D stick figure environment compatible with Brax.

    Represents a 5-segment body:
    - Torso (Root)
    - Left Leg, Right Leg
    - Left Arm, Right Arm

    Restricted to 2D plane (Z-axis constrained).
    """

    def __init__(self, backend="positional"):
        # Define the system config (MJCF-like format)
        # Simplified humanoid: Torso + 4 limbs
        self._config = """
        bodies {
          name: "torso"
          mass: 10.0
          inertia { x: 1.0 y: 1.0 z: 1.0 }
          colliders { capsule { radius: 0.15 length: 0.6 } }
        }
        bodies {
          name: "l_thigh"
          mass: 3.0
          inertia { x: 0.1 y: 0.1 z: 0.1 }
          colliders { capsule { radius: 0.1 length: 0.5 } }
        }
        bodies {
          name: "r_thigh"
          mass: 3.0
          inertia { x: 0.1 y: 0.1 z: 0.1 }
          colliders { capsule { radius: 0.1 length: 0.5 } }
        }
        bodies {
            name: "l_arm"
            mass: 2.0
            inertia { x: 0.1 y: 0.1 z: 0.1 }
            colliders { capsule { radius: 0.08 length: 0.5 } }
        }
        bodies {
            name: "r_arm"
            mass: 2.0
            inertia { x: 0.1 y: 0.1 z: 0.1 }
            colliders { capsule { radius: 0.08 length: 0.5 } }
        }

        joints {
          name: "l_hip"
          parent: "torso"
          child: "l_thigh"
          angle_limit { min: -130 max: 130 }
          parent_offset { x: -0.2 y: -0.3 }
          child_offset { y: 0.25 }
          rotation { z: 90 }
        }
        joints {
          name: "r_hip"
          parent: "torso"
          child: "r_thigh"
          angle_limit { min: -130 max: 130 }
          parent_offset { x: 0.2 y: -0.3 }
          child_offset { y: 0.25 }
          rotation { z: 90 }
        }
        joints {
            name: "l_shoulder"
            parent: "torso"
            child: "l_arm"
            angle_limit { min: -180 max: 180 }
            parent_offset { x: -0.25 y: 0.2 }
            child_offset { y: 0.25 }
            rotation { z: 90 }
        }
        joints {
            name: "r_shoulder"
            parent: "torso"
            child: "r_arm"
            angle_limit { min: -180 max: 180 }
            parent_offset { x: 0.25 y: 0.2 }
            child_offset { y: 0.25 }
            rotation { z: 90 }
        }

        actuators {
            name: "l_hip"
            joint: "l_hip"
            strength: 100.0
            torque: 100.0
        }
        actuators {
            name: "r_hip"
            joint: "r_hip"
            strength: 100.0
            torque: 100.0
        }
        actuators {
            name: "l_shoulder"
            joint: "l_shoulder"
            strength: 50.0
            torque: 50.0
        }
        actuators {
            name: "r_shoulder"
            joint: "r_shoulder"
            strength: 50.0
            torque: 50.0
        }

        gravity { z: -9.8 }
        dt: 0.02
        substeps: 4
        """
        super().__init__(self._config, backend=backend)

    def reset(self, rng):
        qp = self.sys.default_qp()
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, done, zero = jnp.zeros(3)
        metrics = {}
        return envs.State(qp, obs, reward, done, metrics)

    def step(self, state, action):
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        # Simple reward for upright posture (z-axis of torso)
        # In a real training scenario, this would be the difference
        # between simulated pose and target pose.
        pos = qp.pos[0]  # Torso position
        reward = pos[2]  # Maximize height (keep standing)

        return state.replace(qp=qp, obs=obs, reward=reward)

    def _get_obs(self, qp, info):
        """Observe body position and velocities."""
        return jnp.concatenate(
            [qp.pos.ravel(), qp.rot.ravel(), qp.vel.ravel(), qp.ang.ravel()]
        )
