from environment.kuka_env import KukaEnv
from environment.envwrapper import GymEnvironment, getPtClouds
from data_generation.obstacle_datagen import genPtcloudTabletop, genSelfSupervisedData, getPtCloudFromFile
from data_generation.utils import combineDataframes
from training.trainMPNet import TrainerMPNet
from visualize.visualizeMPNet import visualizeMPNetOnDataset


import matplotlib.pyplot as plt
from time import sleep

if __name__ == "__main__":
    
    # stspace = KukaEnv(GUI=True)
    # env = GymEnvironment(stspace)
    # env.reset()
    # getPtClouds(env.stspace)

    # env.setOMPLSovler()
    # sleep(5000)

    
    ######################################################
    ### trining
    # trainer_mpnet = TrainerMPNet(is_MPNET=True)
    # trainer_mpnet.train()

    visualizeMPNetOnDataset(mode='val')
    #####################################################
    # env = GymEnvironment(stspace)
    # env.setOMPLSovler()
    # env.reset(new_obstacle=True)

    # out = env.stspace.get_collision_position()

    # print('before out')
    # print(out)
    # sol = env.planOMPL(env.state)
    # print(f'\nsolution len: {len(sol)}\n')

    # genSelfSupervisedData()

    # genPtcloudTabletop()
    # getPtCloudFromFile()

    # combineDataframes()



    # if sol:
    #     gifs = env.stspace.plot(sol, make_gif=False)
    #     for gif in gifs:
    #         plt.imshow(gif)
    #         plt.show()