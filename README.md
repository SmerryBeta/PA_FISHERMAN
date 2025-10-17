# PA_FISHERMAN
This script is aims to fishing in PartyAnimals automatically.

![image](https://github.com/SmerryBeta/PA_FISHERMAN/blob/main/github_tutorial_img/howtouses.png)

# More details about version
Python 310

Paddlepaddle 2.6.2

Paddleocr 2.9.1

# The statements you need to run in the terminal which can install the dependencies for running this script.

pip install keyboard
pip install pyautogui
pip install pynput
pip install pyyaml
pip install opencv-python
pip install winsound
pip install win32api
pip install Paddlepaddle==2.6.2
pip install Paddleocr==2.9.1

# Other 

If your screen are NOT 1080p, You needs to replace the "you_got.png" from capture your own screen with the same content. Otherwise, the script may can't recognized the new turn of fishing is coming. 
However, the X1~Y2 and F_X1 ~ F_Y2 are also essential for its work. X1~Y2 params are works on recognize how many bails we have, while the  F_X1~F_Y2 params are works on recognize the sometime occured like failure on fishing, fulled bucke, etc.

# The entire process

Start >>(Check bails)>> Cast the fishing rod >>(Check status)>> Waiting for FUCKING 黄鸭叫 >>(Chech status & bails)>> PULL the fishing rod when fish swallowed the bait for PULL_TIME >> 
REALEASE for REALEASE_TIMEE >> Loops with pull for PULL_TIME_LOOP second and realse for RELEASE_TIME_LOOP second until it failed or the fish catched >> Capture the consequences and goes to the next turn.

# What I wants to say

oh~ 猛兽的策划好像没母亲~ 暗改了爆率, 又暗改手感~ (The rhythm of "跳楼机" which is a Chinese song)

<img width="594" height="923" alt="image" src="https://github.com/user-attachments/assets/5fa907cf-37f1-46bf-9733-cb97fa73f508" />
