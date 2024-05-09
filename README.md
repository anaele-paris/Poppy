# Poppy

![poppy](https://github.com/PierreBio/Poppy/assets/45881846/b8e52bd4-a2ce-4758-8147-6c9afbb46cbb)

This project is carried out in the context of the Artificial Intelligence Masters of **TelecomParis & ENSTA** in the "Learning for Robotics" class of Mai NGUYEN & Damien BOUCHABOU.

This project has been conducted by Anaële BAUDANT-COJAN, Pierre BILLAUD and Edouard DUCLOY of the 2023-2024 AI Master. 

<sub>Made with __Python v3.8.10__ and __Coppelia Sim__ </sub>

## Project

Creation of a virtual environment with Coppelia Sim dedicated to make a robot executing some tasks using Reinforcement Learning (RL).

## How to setup?

- First, clone the repository:

```
git clone https://github.com/PierreBio/Poppy.git
```

- Then go to the root of the project:

```
cd Poppy
```

- Install [Python v3.8.10](https://www.python.org/downloads/release/python-3810/) (or use __pyenv__ or __Visual Studio__ to create your environment).

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

- Once the project and environment are setup, you can launch it by opening **sujet_upper_body_imitation.ipynb** with __Visual Studio__ or __Jupyter__.

- (You have to link your **Visual Studio** or **Jupyter** with your created environment using Python v3.8.10)

- Launch Coppelia Sim on your computer on a new scene.

- You can execute now cells from your notebook "src/Poppy_RL.ipynb" to execute code on your Coppelia Sim environement.


__Note: Do not forget to restart Python Kernel after modifying Poppy_Env.__

## Ressources

[Coppelia Sim](https://www.coppeliarobotics.com/)
