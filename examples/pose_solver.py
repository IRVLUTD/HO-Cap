# Python standard lib
import os, sys
import argparse
import numpy as np
import casadi as cs
from typing import List, Dict, Tuple, Union
from transforms3d.quaternions import mat2quat

# OpTaS
from _init_paths import *
import lib.optas as optas
from lib.optas.models import TaskModel
from lib.optas.spatialmath import angvec2r, r2angvec


class SDFTaskModel(TaskModel):

    # robot model from optas
    def __init__(
            self,
            name: str,
            dim: int,
            time_derivs: List[int] = [0],
            symbol: str = "y",
            dlim: Dict[int, Tuple[List[float]]] = {},
            T: Union[None, int] = None,
            is_discrete: bool = False,            
        ):
        
        super().__init__(name, dim, time_derivs, symbol, dlim, T, is_discrete)
        self.field_margin = 0.1
        self.grid_resolution = 0.01


    def setup_points_field(self, points):

        self.workspace_bounds = np.stack((points.min(0), points.max(0)), axis=1)
        margin = self.field_margin
        self.origin = np.array([self.workspace_bounds[0][0] - margin, self.workspace_bounds[1][0] - margin, self.workspace_bounds[2][0] - margin]).reshape((1, 3))
        workspace_points = np.array(np.meshgrid(
                                np.arange(self.workspace_bounds[0][0] - margin, self.workspace_bounds[0][1] + margin, self.grid_resolution),
                                np.arange(self.workspace_bounds[1][0] - margin, self.workspace_bounds[1][1] + margin, self.grid_resolution),
                                np.arange(self.workspace_bounds[2][0] - margin, self.workspace_bounds[2][1] + margin, self.grid_resolution),
                                indexing='ij'))
        self.field_shape = workspace_points.shape[1:]
        self.workspace_points = workspace_points.reshape((3, -1)).T
        self.field_size = self.workspace_points.shape[0]
        print('origin', self.origin)
        print('workspace field shape', self.field_shape)
        print('workspace field', self.field_size)
        print('workspace points', self.workspace_points.shape)


    def points_to_offsets(self, points):
        n = points.shape[0]
        origin = np.repeat(self.origin, n, axis=0)
        idxes = optas.floor((points - origin) / self.grid_resolution)
        idxes[:, 0] = cs.fmax(idxes[:, 0], 0)
        idxes[:, 0] = cs.fmin(idxes[:, 0], self.field_shape[0] - 1)
        idxes[:, 1] = cs.fmax(idxes[:, 1], 0)
        idxes[:, 1] = cs.fmin(idxes[:, 1], self.field_shape[1] - 1)
        idxes[:, 2] = cs.fmax(idxes[:, 2], 0)
        idxes[:, 2] = cs.fmin(idxes[:, 2], self.field_shape[2] - 1)
        # offset = n_3 + N_3 * (n_2 + N_2 * n_1)
        # https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays
        offsets = idxes[:, 2] + self.field_shape[2] * (idxes[:, 1] + self.field_shape[1] * idxes[:, 0])
        return offsets              


class PoseSolver:

    def __init__(self, task_model):
        self.task_model = task_model
        self.task_name = self.task_model.name


    def setup_optimization(self, num_points):
        # Setup optimization builder
        builder = optas.OptimizationBuilder(T=1, tasks=[self.task_model])

        # setup parameters
        # object points for SDF optimization
        object_points = builder.add_parameter("object_points", num_points, 3)
        # sdf field
        sdf_cost = builder.add_parameter("sdf_cost", self.task_model.field_size)
        
        # get robot state variables
        q_T = builder.get_model_states(self.task_name)

        # axis-angle to rotation matrix
        rv = q_T[:3]
        angle = optas.norm_2(rv)
        axis = rv / angle
        R = angvec2r(angle, axis)

        # Setting optimization - cost term and constraints
        points_tf = R @ object_points.T + q_T[3:].reshape((3, 1))
        offsets = self.task_model.points_to_offsets(points_tf.T)
        builder.add_cost_term("sdf_cost", optas.sumsqr(sdf_cost[offsets]))

        # setup solver
        solver_options = {'ipopt': {'max_iter': 50, 'tol': 1e-15}}
        self.solver = optas.CasADiSolver(builder.build()).setup("ipopt", solver_options=solver_options)    


    def solve_pose(self, RT, object_points, sdf_cost):
        rv = r2angvec(RT[:3, :3])
        x0 = np.zeros((6, ), dtype=np.float32)
        x0[:3] = rv
        x0[3:] = RT[:3, 3]
        self.solver.reset_initial_seed({f"{self.task_name}/y/x": x0})

        self.solver.reset_parameters({"sdf_cost": optas.DM(sdf_cost),
                                    "object_points": optas.DM(object_points)})
                  
        solution = self.solver.solve()
        y = solution[f"{self.task_name}/y"]

        print("***********************************") 
        print("Casadi SDF pose solution:")
        print(y, y.shape)
        return y.toarray().flatten()


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-r",
        "--robot",
        type=str,
        default="panda",
        help="Robot name",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = make_args()

    RT = np.array([[-0.05241979, -0.45344928, -0.88973933,  0.41363978],
        [-0.27383122, -0.8502871,   0.44947574,  0.12551154],
        [-0.96034825,  0.26719978, -0.07959669,  0.97476065],
        [ 0.,          0.,          0.,          1.        ]])   
    
    name = 'object_pose_estimator'
    dim = 6
    task_model = SDFTaskModel(name, dim)

    # fake points
    num_points = 100
    points = np.random.randn(num_points, 3)
    task_model.setup_points_field(points)

    # solve problem
    pose_solver = PoseSolver(task_model)
    pose_solver.setup_optimization(num_points)

    sdf_cost = np.random.randn(task_model.field_size)
    y_solution = pose_solver.solve_pose(RT, points, sdf_cost)