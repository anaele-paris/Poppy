# Poppy

![poppy](https://github.com/PierreBio/Poppy/assets/45881846/b8e52bd4-a2ce-4758-8147-6c9afbb46cbb)

This project is carried out in the context of the Artificial Intelligence Masters of **TelecomParis**.

<sub>Made with __Python v3.8__ and __Coppelia Sim__ </sub>

## Project

Creation of a virtual environment with Coppelia Sim dedicated to make a robot executing some tasks.

## How to setup?

- First, clone the repository:

```
git clone https://github.com/PierreBio/Poppy.git
```

- Then go to the root of the project:

```
cd Poppy
```

- Install [Python v3.8](https://www.python.org/downloads/release/python-3810/) (or use __pyenv__ or __Visual Studio__ to create your environment).

- Create a virtual environment using **Python v3.8**:

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

- (You have to link your **Visual Studio** or **Jupyter** with your created environment)

- Launch Coppelia Sim on your computer and open the specific scene __REMOTE_API_TEMPFILE_6057__ located at __./Poppy/scenes/REMOTE_API_TEMPFILE_6057.ttt__.

- You can execute now cells inside your notebook to make some changes on your Coppelia Sim.

## Ressources

[Coppelia Sim](https://www.coppeliarobotics.com/)
