using POMDPs
using Distributions
using POMDPToolbox
using SARSOP

#First, define the struct that will characterize each state. In this case, a state is defined 
#with the current (x,y) tuple of the Agent's location in the grid, along with the(x,y) 
#tuple of the location of the gold.  Multiple reward states (i.e. multiple cells 
#containing gold) could be represented with a vector of tuples.

struct GridState
    xA::Int64
    yA::Int64
    xGold::Int64
    yGold::Int64
    xW::Int64
    yW::Int64
    xP1::Int64
    yP1::Int64
    xP2::Int64
    yP2::Int64
end

#Once created, a GridState constructor is built, in this case containing a pre-set 
#gold location (in cell [2,1]). Note the constructor requires specification only 
#of Agent location for ease.

# , xG::Int64, yG::Int64, xW::Int64, yW::Int64, xP1::Int64, yP1::Int64, xP2::Int64, yP2::Int64
GridState(xA::Int64, yA::Int64) = GridState(xA,yA,2,1,3,1,4,1,3,3) #Arbitrary initializations for G,W,Pits

#To define the terminal state and shape observations, a helper function specifies 
#when the agent "finds" the gold. Discovery is simple: the agent finds and retrieves
#gold when agent and gold are co-located. 

goldFound(s1::GridState) = s1.xA==s1.xGold && s1.yA==s1.yGold
wumpusFound(s1::GridState) = s1.xA==s1.xW && s1.yA==s1.yW
pitFound(s1::GridState) = (s1.xA==s1.P1 && s1.yA==s1.P1) || (s1.xA==s1.P2 && s1.yA==s1.P2)

function wumpusNear(s1::GridState)
    for n in getNeighbors(s1.xA, s1.yA)
        if (n[1] == st.xW) && (n[2] == st.yW)
            return true
        end
    end
    return false
end

function pitNear(s1::GridState)
    for n in getNeighbors(s1.xA, s1.yA)
        if (n[1]==s1.xP1 && n[2]==s1.yP1) || (n[1]==s1.xP2 && n[2]==s1.yP2)
            return true
        end
    end
    return false
end

#Once the foundation of a GridState has been defined, the GridWorld problem itself
#can be characterized. Here, the problem is defined using the size of the world 
#(i.e. the number of grid cells), the reward (r) and the discount factor. Note that the 
#problem can be expanded by adding additional fields including other rewards 
#(or costs as negative rewards), the accuracy level of observations, etc

type GridPOMDP <: POMDP{GridState, Int64, Int64}  # POMDP{S,A,O}
    size_x::Int64       #Number of grid cells in the x-direction
    size_y::Int64       #Number of grid cells in the x-direction
    r_G::Int64          #Reward for finding gold
    r_WP::Int64         #Reward for Wumpus/Pit
    r_S::Int64          #Reward for Shoot
    r_M::Int64          #Reward for move
    discount::Float64   #Discount factor
end

#Once defined, a simple problem constructor specifies the default characteristics. 
#The following specifies a 4x4 grid world, with a reward of +1000 (Gold), etc., and a discount factor.

function GridPOMDP()	
	return GridPOMDP(4,4,1000,-1000,-10,-1,0.95)
end;

#With the problem specified, a problem instance can be built

pomdp = GridPOMDP()

#A paired function establishes that finding the gold makes the state terminal.
POMDPs.isterminal(pomdp::GridPOMDP, s::GridState) = goldFound(s)

######STATES######

#For the state distribution, each state is pushed onto an array and indexed

function POMDPs.states(pomdp::GridPOMDP)
    s = GridState[] #initialize array of GridWorldStates
    for xA=1:pomdp.size_x, yA=1:pomdp.size_y, xGold=1:pomdp.size_x, yGold=1:pomdp.size_y, xW=1:pomdp.size_x, yW=1:pomdp.size_y, xP1=1:pomdp.size_x, yP1=1:pomdp.size_y, xP2=1:pomdp.size_x, yP2=1:pomdp.size_y
        push!(s, GridState(xA,yA,xGold,yGold,xW,yW,xP1,yP1,xP2,yP2))
    end
    return s   #array of states
end;

function POMDPs.state_index(pomdp::GridPOMDP, state::GridState)
    return sub2ind((pomdp.size_x, pomdp.size_y, pomdp.size_x, pomdp.size_y, pomdp.size_x, pomdp.size_y, pomdp.size_x, pomdp.size_y, pomdp.size_x, pomdp.size_y,), state.xA, state.yA, state.xGold, state.yGold, state.xW,state.yW, state.xP1, state.yP1, state.xP2, state.yP2)
end

#The POMDP package requires computation of the number of expected states

POMDPs.n_states(p::GridPOMDP) = (p.size_x*p.size_y)^5  #Agent,G,W,P1,P2

######ACTIONS#######

#The next functions specify general parameters concerning the actions available to the agent.
#In this simple world, the agent can move right, left, up, or down, which are specified in the 
#actions function as integers (1=right, 2=left, 3=up, 4=down). Actions can also be specified 
#as strings (symbols in Julia) as follows: [:right, :left, :up, :down]. The conversion between 
#action representation and action index may be updated in the action_index function below.
#The number of actions must also be explicitly specified in the n_actions function. 

POMDPs.actions(p::GridPOMDP) = [1,2,3,4,5,6]  #R,L,U,D,Shoot,NoOp
POMDPs.n_actions(p::GridPOMDP) = 6
POMDPs.actions(pomdp::GridPOMDP, state::GridState) = POMDPs.actions(pomdp)

#The action_index function enables conversion between the action representation and the 
#action index that will be used to track agent location. So, if the "move right" action was
#a symbol, :right, replace a==1 with a==:right.

function POMDPs.action_index(::GridPOMDP, a::Int64)
    if a==1
        return 1
    elseif a==2
        return 2
    elseif a==3
        return 3
    elseif a==4
        return 4
    elseif a==5
        return 5
	else
        return 6
    end
    error("invalid action: $a")  #note the $ placeholder for var reference in print
end;

######TRANSITION FUNCTION######
#The transition function models how the agent moves through the grid world.
#Function isInbounds helps determine whether a targeted action is possible within
#the bounds of the world and subseuently shapes the state result of actions.

function isInbounds(pomdp::GridPOMDP, st::GridState)
    if (1 <= st.xA <= pomdp.size_x) && (1 <= st.yA <= pomdp.size_y)
        return true
    end
    return false
end

function getNeighbors(x::Int64, y::Int64)
    # Refactored from transition()
    #The neighbor array represents the possible states to which the
    #agent in its current state may transition. The states correspond to 
    #the integer representation of each action.
    neighbor = [
        (x+1,y),  #right
        (x-1,y),  #left
        (x,y+1),  #up
        (x,y-1),   #down
        (x,y),       #original cell
        (x,y),       #original cell
    ]
    return neighbor
end
           
            
#The transition function uses the isInbounds function to determine where an action
#delivers the agent. If the targeted action is out of bounds, the 
#agent rebounds into the original cell.

function POMDPs.transition(p::GridPOMDP, s::GridState, a::Int64)
    # Refactored neighbor array to global scope, for use with observation()
    
    #The target cell is the location at the index of the appointed action.
    target = getNeighbors(s.xA, s.yA)[a]
    s.xA = target[1]
    s.yA = target[2]
    target = s  # Seems index in Julia is 1-based.
	
	#If the target cell is out of bounds, the agent remains in 
	#the same cell. Otherwise the agent transitions to the target 
	#cell.
    if !isInbounds(p,target)	
		return SparseCat([s], [1.0])
	else
		return SparseCat([target], [1.0])
	end
end

######OBSERVATIONS######
#Like actions and states, observations are specified both through explicit parameters and 
#through distributions. This simple implementation uses just one type of observation: "glitter",
#the presence of gold in the agent's current cell location; correspondingly a simple binary 
#observation structure is used: an observation of glitter corresponds to a "true" observation, 
#which is produced when the agent is co-located with gold. The range of observations may be 
#expanded by modifying this representation using an equally simple approach, for example, using 
#integers to represent the presence of each type of observation.

POMDPs.observations(::GridPOMDP) = [1,2,3]  #Glitter, Stench, Breeze
POMDPs.observations(pomdp::GridPOMDP, s::GridState) = POMDPs.observations(pomdp);
POMDPs.n_observations(::GridPOMDP) = 3

#The observation distribution establishes the likelihood
#of a true observation (glitter)
type ObservationDistribution
    op_glitter::Float64
    op_stench::Float64
    op_breeze::Float64  # Can get from 1-rest?
end
ob_dist = 1.0/16
ObservationDistribution() = ObservationDistribution(ob_dist)
iterator(od::ObservationDistribution) = [1,2,3]

#The observation function and density function maintain the observations 
#received by the agent. The density function (pdf) establishes the value 
#of the distribution at a particular sample. The observation function (below)
#determines the likelihood of an observation at a particular state.

function POMDPs.pdf(od::ObservationDistribution, obs::Int64)
	if obs==1	
		return od.op_glitter
	elseif obs==2
		return od.op_stench
    elseif obs==3
        return 1 - (od.op_glitter + od.op_stench)  # TODO:What's correct?
	end
end

#Sampling function for use in simulation

function POMDPs.rand(rng::AbstractRNG, od::ObservationDistribution)
	if rand(rng) <= od.op_glitter
		return 1
	elseif od.op_glitter < rand(rng) <= od.op_stench 	
		return 2
    else
        return 3
	end
end

function POMDPs.observation(pomdp::GridPOMDP, s::GridState)
	od = ObservationDistribution()
                    
	if goldFound(s)
		od.op_glitter = 1.0
	else
		od.op_glitter = 0.0
	end
    
    if wumpusNear(s)
		od.op_stench = 1.0
	else
		od.op_stench = 0.0
	end
    
    if pitNear(s)
		od.op_breeze = 1.0
	else
		od.op_breeze = 0.0
	end
	od
end

#The reward function tracks the current reward, in this case by adding to the reward sum 
#if gold has been found and then returning the current total. 
#The function can be expanded with other rewards/costs by including those 
#additions/subtractions in other conditional branches 

function POMDPs.reward(p::GridPOMDP, s::GridState, a::Int64)
	r = 0.0
	if goldFound(s)
		r +=  p.r_G
	elseif wumpusFound(s) || pitFound(s)
		r += p.r_WP
    elseif a == 5
        r += p.r_S
    elseif a == 6  # NoOp
        r += 0.0
    else
        r += p.r_M
	end
	r
end

POMDPs.reward(pomdp::GridPOMDP, s::GridState, a::Int64, obs::Bool) = reward(pomdp,s,a)

POMDPs.discount(p::GridPOMDP) = p.discount

#The initial state distribution establishes the initial distribution over states.
#A SparseCat sparse array is used given the few states that have a non-zero 
#likelihood of occupancy.

function POMDPs.initial_state_distribution(pomdp::GridPOMDP) 
    return SparseCat([GridState(1,1)], [1.0])  # An arbitrary initialization
end;

