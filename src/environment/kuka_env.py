import numpy as np
import pybullet as p
from time import sleep
import pybullet_data
import pickle
from time import time
try:
    from environment.timer import Timer
except:
    from timer import Timer
    
import matplotlib.pyplot as plt

class KukaEnv:
    '''
    Interface class for maze environment
    '''
    RRT_EPS = 0.5

    def __init__(self, GUI=False, kuka_file="src/environment/kuka_iiwa/model_0.urdf"):
        # print("Initializing environment...")

        self.kuka_file = kuka_file

        self.collision_check_count = 0
        self.collision_time = 0
        self.collision_point = None
        self.collision_info = []
        self.invisibile_objectIDs = []

        if GUI:
            p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
        else:
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition = [0, 0, 0.1])

        self.timer = Timer()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.kukaId = p.loadURDF(kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        p.performCollisionDetection()

        self.tableId = p.loadURDF("table/table.urdf", [0, 0, -.6], [0, 0, 0, 1], useFixedBase=True)
        self.table2Id = p.loadURDF("table/table.urdf", [0, 1, -.6], [0, 0, 0, 1], useFixedBase=True)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, -.6], [0, 0, 0, 1], useFixedBase=True)


        self.config_dim = p.getNumJoints(self.kukaId)

        if self.config_dim == 7:
            # width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera()

            camera_target_position = [0, .5, 0]
            cam_yaw = 45
            cam_pitch = -45
            cam_dist = 1.75
            
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_dist,
                cameraYaw=cam_yaw,
                cameraPitch=cam_pitch,
                cameraTargetPosition=camera_target_position)

            self.projectionMatrix = p.computeProjectionMatrixFOV(
                                    fov=55.0,
                                    aspect=1.0,
                                    nearVal=0.1,
                                    farVal=8.0
                                    )

            self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=camera_target_position,
                distance=cam_dist,
                yaw= cam_yaw,
                pitch=cam_pitch,
                roll=0,
                upAxisIndex = 2
            )
            # # alternate way to represent camera
            # self.viewMatrix = p.computeViewMatrix(
            #     cameraEyePosition=[2, 2, 2],
            #     cameraTargetPosition=camera_target_position,
            #     cameraUpVector= [0,0,1]
            # )
            
            self.viewMatrix_2 = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=camera_target_position,
                distance=cam_dist,
                yaw= -45,
                pitch=cam_pitch,
                roll=0,
                upAxisIndex = 2
            )


        self.pose_range = [(p.getJointInfo(self.kukaId, jointId)[8], p.getJointInfo(self.kukaId, jointId)[9]) for
                           jointId in range(p.getNumJoints(self.kukaId))]
        self.bound = np.array(self.pose_range).T.reshape(-1)

        self.kukaEndEffectorIndex = self.config_dim - 1

        p.setGravity(0, 0, -10)
    
    def disconnect(self):
        p.disconnect()
    
    def get_collision_position(self):
        if self.collision_info:
            col_arr = np.array(self.collision_info)
            dists = np.linalg.norm(col_arr, axis=1)
            return [col_arr[np.argmin(dists)], col_arr[np.argmax(dists)]]
        else:
            return []

    def get_links_position(self, config=None):
        '''Gets lines in the center of the robot links
        '''
        points = []
        if config is not None:
            for i in range(p.getNumJoints(self.kukaId)):
                p.resetJointState(self.kukaId, i, config[i])
        for effector in range(self.kukaEndEffectorIndex + 1):
            point = p.getLinkState(self.kukaId, effector)[4]
            point = (point[0], point[1], point[2])
            points.append(point)
        return points

    def get_point_cloud(self, width, height, view_matrix, proj_matrix):
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

        # get a depth image
        # "infinite" depths will have a value close to 1
        _, _, rgbaImg, depth, segImg = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)

        def changeTransparency(objid, opacity=.5):
            for data in p.getVisualShapeData(objid):
                color = list(data[-1])
                color[-1] = opacity
                p.changeVisualShape(objid, data[1], rgbaColor=color)
        objidx = [self.planeId, self.tableId, self.table2Id, self.kukaId]
        objidx.extend(self.invisibile_objectIDs)
        for idx in objidx:
            changeTransparency(idx, 0)

        _, _, _, depth, _ = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)

        for idx in objidx:
            changeTransparency(idx, 1.0)
        
        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        if False: # For debugging
            n=2800
            idx = np.random.choice(range(points.shape[0]), n, replace=False)
            colors = np.tile([1, 0, 0], (n,1))
            p.addUserDebugPoints(pointPositions=points[idx,:], pointColorsRGB=colors, pointSize=2.0, lifeTime=0)

        return points, rgbaImg, segImg

    def __str__(self):
        return 'kuka'+str(self.config_dim)
        
    def remove_body(self, bodyid):
        p.removeBody(bodyid)

    def set_random_init_goal(self, dist=0, dist_worksp=0, constraint_fn=None):
        self.init_state = None
        self.goal_state = None
        while (self.init_state is None) or (self.goal_state is None):
            points = self.sample_n_points(n=2)
            init = points[0] if self.init_state is None else init
            goal = points[1] if self.goal_state is None else goal

            worksp_1 = np.asarray(self.get_robot_points(init))
            worksp_2 = np.asarray(self.get_robot_points(goal))
            if np.sum(np.abs(init - goal)) > dist and np.linalg.norm(worksp_2 - worksp_1) > dist_worksp:
                if constraint_fn is None:
                    self.init_state, self.goal_state = init, goal
                else:
                    if constraint_fn(worksp_1):
                        self.init_state = init
                    if constraint_fn(worksp_2):
                        self.goal_state = goal

    def get_robot_points(self, config=None, end_point=True):
        points = []
        if config is not None:
            for i in range(p.getNumJoints(self.kukaId)):
                p.resetJointState(self.kukaId, i, config[i])
        if end_point:
            point = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
            point = (point[0], point[1], point[2])
            return point
        for effector in range(self.kukaEndEffectorIndex + 1):
            point = p.getLinkState(self.kukaId, effector)[0]
            point = (point[0], point[1], point[2])
            points.append(point)
        return points

    def create_voxel(self, halfExtents, basePosition):
        groundColId = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        groundVisID = p.createVisualShape(shapeType=p.GEOM_BOX,
                                          rgbaColor=np.random.uniform(0, 1, size=3).tolist() + [0.8],
                                          specularColor=[0.4, .4, 0],
                                          halfExtents=halfExtents)
        groundId = p.createMultiBody(baseMass=0,
                                     baseCollisionShapeIndex=groundColId,
                                     baseVisualShapeIndex=groundVisID,
                                     basePosition=basePosition)
        return groundId

    def sample_n_points(self, n, need_negative=False):
        if need_negative:
            negative = []
        samples = []
        for i in range(n):
            while True:
                sample = self.uniform_sample()
                if self._state_fp(sample):
                    samples.append(sample)
                    break
                elif need_negative:
                    negative.append(sample)
        if not need_negative:
            return samples
        else:
            return samples, negative

    def uniform_sample(self, n=1):
        '''
        Uniformlly sample in the configuration space
        '''
        self.timer.start()
        sample = np.random.uniform(np.array(self.pose_range)[:, 0], np.array(self.pose_range)[:, 1], size=(n, self.config_dim))
        if n==1:
            self.timer.finish(Timer.SAMPLE)
            return sample.reshape(-1)
        else:
            self.timer.finish(Timer.SAMPLE)
            return sample

    def distance(self, from_state, to_state):
        '''
        Distance metric
        '''
        to_state = np.maximum(to_state, np.array(self.pose_range)[:, 0])
        to_state = np.minimum(to_state, np.array(self.pose_range)[:, 1])
        diff = np.abs(to_state - from_state)

        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def interpolate(self, from_state, to_state, ratio):
        diff = to_state - from_state

        new_state = from_state + diff * ratio
        new_state = np.maximum(new_state, np.array(self.pose_range)[:, 0])
        new_state = np.minimum(new_state, np.array(self.pose_range)[:, 1])

        return new_state

    def in_goal_region(self, state):
        '''
        Return whether a state(configuration) is in the goal region
        '''
        return self.distance(state, self.goal_state) < self.RRT_EPS and \
               self._state_fp(state)

    def set_config(self, c, kukaId=None):
        if kukaId is None:
            kukaId = self.kukaId
        for i in range(p.getNumJoints(kukaId)):
            p.resetJointState(kukaId, i, c[i])
        p.performCollisionDetection()

    def plot(self, path, make_gif=False):
        path = np.array(path)

        self.set_config(path[0])
        prev_pos = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]

        target_kukaId = self.kukaId
        self.set_config(path[-1], target_kukaId)
        final_pos = p.getLinkState(target_kukaId, self.kukaEndEffectorIndex)[0]

        # for data in p.getVisualShapeData(new_kuka):  # change transparency
            #     color = list(data[-1])
            #     color[-1] = 0.5
            #     p.changeVisualShape(new_kuka, data[1], rgbaColor=color)

        gifs = []
        current_state_idx = 0

        while True:
            new_kuka = self.kukaId
            disp = path[current_state_idx + 1] - path[current_state_idx]
            d = self.distance(path[current_state_idx], path[current_state_idx + 1])
            K = int(np.ceil(d / 0.2))
            for k in range(0, K):
                c = path[current_state_idx] + k * 1. / K * disp
                self.set_config(c, new_kuka)
                new_pos = p.getLinkState(new_kuka, self.kukaEndEffectorIndex)[0]
                p.addUserDebugLine(prev_pos, new_pos, [1, 0, 0], 10, 0)
                prev_pos = new_pos
                p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                if make_gif:
                    gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])
            p.loadURDF("cube.urdf", prev_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES) # visualize vertex

            current_state_idx += 1
            if current_state_idx == len(path) - 1:
                self.set_config(path[-1], new_kuka)
                p.addUserDebugLine(prev_pos, final_pos, [1, 0, 0], 10, 0)
                p.loadURDF("sphere2red.urdf", final_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                break

        return gifs

    # =====================internal collision check module=======================

    def _valid_state(self, state):
        return (state >= np.array(self.pose_range)[:, 0]).all() and \
               (state <= np.array(self.pose_range)[:, 1]).all()

    def _point_in_free_space(self, state):
        self.collision_info = []
        t0 = time()
        if not self._valid_state(state):
            return False

        for i in range(p.getNumJoints(self.kukaId)):
            p.resetJointState(self.kukaId, i, state[i])
        p.performCollisionDetection()
        collision_data = p.getContactPoints(bodyA=self.kukaId)
        if len(collision_data) == 0:
            self.collision_check_count += 1
            self.collision_time += time() - t0
            return True
        else:
            self.collision_point = state
            self.collision_check_count += 1
            self.collision_time += time() - t0

            self.collision_info = [it[5] for it in collision_data]
            # print('\n',self.collision_info)
            # arr = np.array(self.collision_info)
            # dists = np.linalg.norm(arr, axis=1)
            # print(dists)
            # idx = np.argmin(dists)
            # p.loadURDF("sphere2red.urdf", arr[idx], globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
            # sleep(500)
            return False

    def _state_fp(self, state):
        self.timer.start()
        free = self._point_in_free_space(state)
        self.timer.finish(Timer.VERTEX_CHECK)
        return free

    def _iterative_check_segment(self, left, right):
        if np.sum(np.abs(left - left)) > 0.1:
            mid = (left + right) / 2.0
            self.k += 1
            if not self._state_fp(mid):
                self.collision_point = mid
                return False
            return self._iterative_check_segment(left, mid) and self._iterative_check_segment(mid, right)

        return True

    def _edge_fp(self, state, new_state):
        self.timer.start()
        self.k = 0
        assert state.size == new_state.size

        if not self._valid_state(state) or not self._valid_state(new_state):
            self.timer.finish(Timer.EDGE_CHECK)
            return False
        if not self._point_in_free_space(state) or not self._point_in_free_space(new_state):
            self.timer.finish(Timer.EDGE_CHECK)
            return False

        disp = new_state - state

        d = self.distance(state, new_state)
        K = int(d / self.RRT_EPS)
        for k in range(0, K):
            c = state + k * 1. / K * disp
            if not self._point_in_free_space(c):
                self.timer.finish(Timer.EDGE_CHECK)
                return False
        self.timer.finish(Timer.EDGE_CHECK)
        return True
