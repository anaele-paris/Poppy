# Poppy

![poppy](https://github.com/PierreBio/Poppy/assets/45881846/b8e52bd4-a2ce-4758-8147-6c9afbb46cbb)

This project is carried out in the context of the Artificial Intelligence Masters of **TelecomParis & ENSTA** in the "Learning for Robotics" class of Mai NGUYEN & Damien BOUCHABOU.

This project has been conducted by Anaële BAUDANT-COJAN, Pierre BILLAUD and Edouard DUCLOY of the 2023-2024 AI Master. 

<sub>Made with __Python v3.8.10__ and __Coppelia Sim__ </sub>

## Project

Creation of a virtual environment with Coppelia Sim dedicated to make a robot execute some mooves following example provided in a video, using Reinforcement Learning (RL).

## How to setup?

- First, clone the repository:

```
git clone git@github.com:anaele-paris/Poppy.git
```

- Then go to the root of the project:

```
cd Poppy
```

- Install [Python v3.8.10](https://www.python.org/downloads/release/python-3810/) (or use __pyenv__ or __Visual Studio__ to create your environment). the pypot library uppon which Poppy is built does not support more recent versions of Python.

- Create a virtual environment using **Python v3.8.10**:

```
py -m venv venv
```

- Activate your environment:

```
.\venv\Scripts\activate
```

- Install the requirements:

```
py -m pip install -r requirements.txt
```

## How to launch?

- Once the project and environment are setup, you can launch it by opening **Poppr_RL.ipynb** to train the models with __Visual Studio__ or __Jupyter__. Alternatively, to better understand the projet, you can open the following files in ./Ressources:
- **tuto_poppy_coppelia.ipynb** to explore how Poppy can be activated in Coppelia Simulator (using pypot library).
- **tuto_video_capturing.ipynb** to capture video and extract the Poppy skeletons with will be used as target for the Reinforcement Learning (RL), using the Blase Pose pre-trained model (with mediapipe library) for pose estimation and preprocessing functions (in utils).
- **sujet_upper_body_estimation_ENSTA.ipynb** the instructions we were provided with during our "Learning for Robotics" class at ENSTA.

- (You have to link your **Visual Studio** or **Jupyter** with your created environment using Python v3.8.10)

- Launch the simulator "Coppelia Sim" on your computer, a new (empty) scene will appear. 

- You can execute now cells from your notebook "src/Poppy_RL.ipynb" or "tuto_poppy_coppelia.ipynb" to execute code on your Coppelia Sim environement. The line "env = PoppyEnv()" will initiate a connection with Coppelia Sim and the Poppy robot will appear on a table in Coppelia Sim. 
    - The line "env.reset()" will reset the robot to its initial position. 
    - The line "env.step(action)" will make the robot execute the action provided in the argument. action is an array corresponding to the angles of the motors of the robot.
    - The line "env.close()" will close the connection with Coppelia Sim and the Poppy robot.


__Note: Do not forget to restart Python Kernel after modifying Poppy_Env.__

## Ressources

[Coppelia Sim](https://www.coppeliarobotics.com/)
[Poppy](https://www.poppy-project.org/)
[Blaze Pose / Mediapipe](https://mediapipe.dev/)
[pypot](https://pypi.org/project/pypot/)