# RL Final Project
To run the eval loop all you have to do is call CNN_InAreaOnly.py and it should just start an evaluation of a trained model. If you want to train you will have to go into the CNN_InAreaOnly.py and uncomment the train commands at the end of the file. To run simply call:

```python3 CNN_InAreaOnly.py```

Once it starts running it will take a while to start up, this is normal as all the grids are being created in the environment. A progress bar should start up for an episode and complete when it achieves 85% completion. You will see an output reward. The current eval loop takes in the GDC1 map.
