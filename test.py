import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# --- 1. The Environment (Unchanged) ---
class World:
    def __init__(self, size=20):
        self.size=size;self.EMPTY,self.WALL,self.FOOD=0,1,2
        self.grid=np.full((size,size),self.EMPTY);self.grid[0,:],self.grid[-1,:],self.grid[:,0],self.grid[:,-1]=self.WALL,self.WALL,self.WALL,self.WALL
        for _ in range(size*2):self.grid[np.random.randint(1,size-2),np.random.randint(1,size-2)]=self.WALL
        for _ in range(40):
            pos=(np.random.randint(1,size-2),np.random.randint(1,size-2))
            if self.grid[pos]==self.EMPTY:self.grid[pos]=self.FOOD
        print("World v15 (The Planner) initialized.")
    def get_view(self,pos,s=5):
        x,y=pos;p=s//2;v=np.full((s,s),self.WALL)
        for i in range(s):
            for j in range(s):
                wx,wy=x-p+i,y-p+j
                if 0<=wx<self.size and 0<=wy<self.size:v[i,j]=self.grid[wx,wy]
        return torch.tensor(v,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    def perform_action(self,pos,a_id):
        acts=[(-1,0),(1,0),(0,-1),(0,1),(0,0)];dx,dy=acts[a_id];new_pos=(pos[0]+dx,pos[1]+dy)
        out="MOVED";
        if not(0<new_pos[0]<self.size-1 and 0<new_pos[1]<self.size-1)or self.grid[new_pos]==self.WALL:new_pos=pos;out="BUMPED_WALL"
        if a_id==4:out="ATE_FOOD"if self.grid[pos]==self.FOOD else"ATE_NOTHING"
        return new_pos,out

# --- 2. Cognitive Components (Unchanged) ---
class VisionNet(nn.Module):
    def __init__(self):super().__init__();self.conv=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,16,3,padding=1),nn.ReLU());self.fc=nn.Linear(16*5*5,32)
    def forward(self,x):x=self.conv(x);x=x.view(x.size(0),-1);return torch.relu(self.fc(x))
class MotorNet(nn.Module):
    def __init__(self,strategy_dim=3):super().__init__();self.fc1=nn.Linear(32+strategy_dim,64);self.fc2=nn.Linear(64,5)
    def forward(self,vt,sv):return self.fc2(torch.relu(self.fc1(torch.cat((vt,sv.unsqueeze(0)),dim=1))))

# --- 3. The Planner Mind ---
class GroundedMind_v15:
    def __init__(self, genome):
        print("GroundedMind v15 initializing...");self.genome=genome
        self.vision=VisionNet();self.motor=MotorNet(strategy_dim=3)
        self.params=list(self.vision.parameters())+list(self.motor.parameters());self.optimizer=optim.Adam(self.params,lr=self.genome['lr'])
        self.alpha,self.beta,self.t_index_threshold=genome['alpha'],genome['beta'],genome['t_index_threshold']
        self.stomach=100.0;self.food_eaten_count=0;self.last_food_pos=None;self.food_trace_decay=0
        self.global_tint=0.0;self.last_total_S=0.0;self.Tindex=0
        
        # Planning parameters
        self.plan_horizon = 4
        self.plan_sims = 16

    def learn(self,loss):
        if loss is None:return
        self.optimizer.zero_grad();loss.backward();self.optimizer.step()

    def pfc_director(self, S_hunger, world):
        # The PFC's main job is now just to set the high-level strategy context
        if self.last_food_pos and world.grid[self.last_food_pos]!=world.FOOD:
            self.food_trace_decay+=1
            if self.food_trace_decay > 10:self.last_food_pos=None;self.food_trace_decay=0
        else:self.food_trace_decay=0
        
        if S_hunger > 0.6 and self.last_food_pos:
            return "GoToFood", torch.tensor([0.0,1.0,0.0])
        return "Exploring", torch.tensor([1.0,0.0,0.0])

    # === THE "MIND'S EYE" - MENTAL SIMULATION ===
    def get_action_via_planning(self, world, pos, strategy_vec, S_hunger):
        best_path_first_action = -1
        lowest_path_stress = float('inf')

        with torch.no_grad():
            for _ in range(self.plan_sims):
                sim_pos = pos
                path_stress = 0
                first_action = -1
                
                for step in range(self.plan_horizon):
                    # In this imagined future, what would I see and how would I act?
                    sim_view = world.get_view(sim_pos)
                    sim_vision_thought = self.vision(sim_view)
                    sim_motor_logits = self.motor(sim_vision_thought, strategy_vec)
                    sim_action_id = torch.multinomial(torch.softmax(sim_motor_logits, dim=1), 1).item()
                    
                    if step == 0: first_action = sim_action_id
                    
                    # What is the predicted consequence?
                    sim_next_pos, sim_outcome = world.perform_action(sim_pos, sim_action_id)
                    sim_motor_S = 1.0 if sim_outcome == "BUMPED_WALL" else 0.0
                    
                    # The predicted stress of this future step
                    path_stress += S_hunger + sim_motor_S
                    sim_pos = sim_next_pos
                
                if path_stress < lowest_path_stress:
                    lowest_path_stress = path_stress
                    best_path_first_action = first_action
        
        return best_path_first_action if best_path_first_action != -1 else random.randint(0,4)

    def live_a_moment(self,world,pos,t):
        self.stomach-=0.2
        S_hunger=max(0,(100.0-self.stomach)/100.0)**2
        
        dominant_strategy, strategy_vec = self.pfc_director(S_hunger, world)
        
        action_id = self.get_action_via_planning(world, pos, strategy_vec, S_hunger)
        
        new_pos,outcome=world.perform_action(pos,action_id)
        
        if outcome=="ATE_FOOD":
            world.grid[pos]=world.EMPTY;self.stomach=100.0;self.food_eaten_count+=1
            print(f"SUCCESS! Food eaten at {pos} at time {t}.");self.last_food_pos=pos

        S_motor=1.0 if outcome=="BUMPED_WALL" else 0.0
        actual_S=S_hunger+S_motor
        
        delta_S=abs(actual_S-self.last_total_S)
        load=(self.alpha*delta_S)+(self.beta*actual_S);self.global_tint+=1.0/(1.0+load);self.last_total_S=actual_S
        
        if self.global_tint>=self.t_index_threshold:
            self.global_tint=0;self.Tindex+=1
            print(f"--- TICK {self.Tindex} @ {t}: Strategy='{dominant_strategy}', S_Total={actual_S:.2f} ---")
            
            reward=2.0 if outcome=="ATE_FOOD" else(-1.0 if S_motor>0 else 0.0)
            if reward!=0:
                # Learning is now direct and grounded in real, not imagined, outcomes
                view_for_learning = world.get_view(pos)
                vision_thought_for_learning = self.vision(view_for_learning)
                motor_logits_for_learning = self.motor(vision_thought_for_learning, strategy_vec)
                loss=-torch.log_softmax(motor_logits_for_learning,dim=1).squeeze()[action_id]*reward
                self.learn(loss)
        return new_pos

# --- Main Simulation Loop ---
zen_survivor_genome={'alpha':0.4,'beta':0.7,'t_index_threshold':18.0,'lr':0.001}
world=World();mind=GroundedMind_v15(zen_survivor_genome)
agent_pos=(world.size//2,world.size//2);max_steps=1500;food_history=[]
print(f"\n--- Running Final Simulation: The Planner ---")
for t in range(max_steps):
    agent_pos=mind.live_a_moment(world,agent_pos,t)
    food_history.append(mind.food_eaten_count)
    if mind.stomach<=0:print(f"\n!!!!!! AGENT DIED OF STARVATION at {t}. !!!!!!");break
if t==max_steps-1:print(f"\n\n****** AGENT SURVIVED THE ENTIRE SIMULATION! ******")
print(f"\n--- Simulation Finished ---");print(f"Agent survived for {t+1} steps. Total Food Eaten: {mind.food_eaten_count}.")

plt.figure(figsize=(12,6));plt.plot(food_history,'g-');plt.title("Proof of Planning & Foresight: Food Eaten Over Time")
plt.xlabel("Time Step");plt.ylabel("Cumulative Food Eaten");plt.grid(True);plt.show()
