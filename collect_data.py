import os

from minicrawl.data_collection.collect_data import collect_data


AGREEMENT = """
 MiniCrawl: a minimalistic dungeon crawler for reinforcement learning and 
 imitation learning research 
 Organizer: Federico Malato (federico.malato@uef.fi
            School of Computing, University of Eastern Finland
 ------------------------------------------------------------------------------
 BACKGROUND

 Purpose of this study is two-fold:
     1) Establish a human-normalized baseline of the environment
     2) Gather a dataset of demonstrations for imitation learning

 What data will be collected and how:
     In these experiments, you will explore a dungeon with 20 levels of depth.
     Your objective is to go as deep as you can. If you fail one level, the
     game will restart.

     All data will be anonymous and no private information will be
     collected.

 How this data will be used:
     The recorded data will be used to...

     1) ... evaluate the quality and the difficulty of the environment.
     2) ... provide a core dataset for future studies.

     This data may be released to the public.

     This data may be used in future research.

 Requirements:
     - A keyboard
     - 20 minutes of uninterrupted time (you can not pause the experiment)

 ------------------------------------------------------------------------------
 AGREEMENT

 By agreeing you:
     - ... confirm you recognize what data will be collected and how
       it will be used (scroll up).
     - ... fulfill the requirements (see above).
     - ... acknowledge there is no compensation.
     - ... acknowledge your participation is voluntary and you may withdraw at
       any point before end of the experiment.

 Input 'I agree' to continue, "quit" to close this software."""

INSTRUCTIONS = """
 INSTRUCTIONS

     This experiment will take 20 minutes, and consists
     of four 5-minute games of Doom (1993).

     Your goal is to reach the end of the dungeon.
     To reach the next floor, navigate the maze and collect the key.
     
     Some levels of the dungeon will require a different skill. Follow
     the instructions on the wall and complete the challenge to advance.

 BUTTONS

     WASD:             Forward, left, backward and right
     E:                Pick up
     Q:                Drop

 Press ENTER to start the game"""

AFTER_GAMES = """
 Games finished.
 [NOTE] There is no data uploading in shared code.
 
 Please zip your results folder and send it to federico.malato@uef.fi, or upload
 your results at: https://drive.google.com/drive/u/0/folders/1LjKhT1kNy5c73vAf08IKdCtWJdrVM6LB
 
 Thank you for participating!
"""

NUM_GAMES = 2


if __name__ == '__main__':
    os.system("cls")
    print(AGREEMENT)
    agreement = ""
    while agreement not in ["i agree", "quit"]:
        agreement = input(" >>> ").lower()
    if agreement == "quit":
        exit(0)

    os.system("cls")
    print(INSTRUCTIONS)
    input()

    collect_data(NUM_GAMES)

    print(AFTER_GAMES)
