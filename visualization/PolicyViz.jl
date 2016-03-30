module PolicyViz

export viz_policy, get_belief,get_qval,Policy,read_policy,evaluate
using GridInterpolations, Interact, PGFPlots, Colors, ColorBrewer


const RANGEMAX = 62000.0 #ft
const ranges = [499, 800, 2000, 3038, 5316, 6450, 7200, 7950, 8725, 10633, 13671, 16709, 19747, 
                22785, 25823, 28862, 31900, 34938, 37976, 41014, 48608, 60760]
const thetas = linspace(-pi,pi,41)
const psis   = linspace(-pi,pi,41)
const sos    = [100, 200, 300, 400, 500, 600, 700, 800]
const sis    = [0, 100, 200, 300, 400, 500, 600, 700, 800]
const taus   = [0, 1, 5, 10, 20, 40, 60, 80, 100]
const pas    = [1, 2, 3, 4, 5]
const pasTrue = [0, 1.5, -1.5, 3.0, -3.0]
const NSTATES = length(ranges)*length(thetas)*length(psis)*length(sos)*length(sis)*length(taus)*length(pas)
const ACTIONS = deg2rad([0.0 1.5 -1.5 3.0 -3.0])



type Policy
    alpha       :: Matrix{Float64}
    actions     :: Matrix{Float64}
    nactions    :: Int64
    qvals       :: Vector{Float64}

    function Policy(alpha::Matrix{Float64}, actions::Matrix{Float64})
        return new(alpha, actions, size(actions, 2), zeros(size(actions, 2)))
    end # function Policy
end

#Julia implementation of "load_network" function
type NNet
    file::AbstractString
    weights::Array{Any,1}
    biases::Array{Any,1}
    symmetric::Int32
    numLayers::Int32
    inputSize::Int32
    outputSize::Int32
    maxLayerSize::Int32
    
    layerSizes::Array{Int32,1}
    mins::Array{Float64,1}
    maxes::Array{Float64,1}
    means::Array{Float64,1}
    ranges::Array{Float64,1}
    
    function NNet(file::AbstractString)
        this  = new()
        this.file = file
        f = open(this.file)
        line = readline(f)
        line = readline(f)
        record = split(line,[',','\n'])
        this.numLayers = parse(Int32,record[1])
        this.inputSize = parse(Int32,record[2])
        this.outputSize = parse(Int32,record[3])
        this.maxLayerSize=parse(Int32,record[4])
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.layerSizes = zeros(this.numLayers+1)
        for i=1:(this.numLayers+1)
            this.layerSizes[i]=parse(Int32,record[i])
        end
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.symmetric = parse(Int32,record[1])
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.mins = zeros(this.inputSize)
        for i=1:(this.inputSize)
            this.mins[i]=parse(Float64,record[i])
        end
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.maxes = zeros(this.inputSize)
        for i=1:(this.inputSize)
            this.maxes[i]=parse(Float64,record[i])
        end
        
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.means = zeros(this.inputSize+1)
        for i=1:(this.inputSize+1)
            this.means[i]=parse(Float64,record[i])
        end
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.ranges = zeros(this.inputSize+1)
        for i=1:(this.inputSize+1)
            this.ranges[i]=parse(Float64,record[i])
        end
        
        
        this.weights = Any[zeros(this.layerSizes[2],this.layerSizes[1])]
        this.biases  = Any[zeros(this.layerSizes[2])]
        for i=2:this.numLayers
            this.weights = [this.weights;Any[zeros(this.layerSizes[i+1],this.layerSizes[i])]]
            this.biases  = [this.biases;Any[zeros(this.layerSizes[i+1])]]
        end
        
        layer=1
        i=1
        j=1
        line = readline(f)
        record = split(line,[',','\n'])
        while !eof(f)
            while i<=this.layerSizes[layer+1]
                while record[j]!=""
                    this.weights[layer][i,j] = parse(Float64,record[j])
                    j=j+1
                end
                j=1
                i=i+1
                line = readline(f)
                record = split(line,[',','\n'])
            end
            i=1
            while i<=this.layerSizes[layer+1]
                this.biases[layer][i] = parse(Float64,record[1])
                i=i+1
                line = readline(f)
                record = split(line,[',','\n'])
            end
            layer=layer+1
            i=1
            j=1
        end
        close(f)
        
        return this
    end
end

#Evaluates one set of inputs
function evaluate_network(nnet::NNet,input::Array{Float64,1})
    numLayers = nnet.numLayers
    inputSize = nnet.inputSize
    outputSize = nnet.outputSize
    symmetric = nnet.symmetric
    biases = nnet.biases
    weights = nnet.weights
    
    inputs = zeros(inputSize)
    for i = 1:inputSize
        if input[i]<nnet.mins[i]
            inputs[i] = (nnet.mins[i]-nnet.means[i])/nnet.ranges[i]
        elseif input[i] > nnet.maxes[i]
            inputs[i] = (nnet.maxes[i]-nnet.means[i])/nnet.ranges[i] 
        else
            inputs[i] = (input[i]-nnet.means[i])/nnet.ranges[i] 
        end
    end
    if symmetric ==1 && inputs[2]<0
        inputs[2] = -inputs[2]
        inputs[1] = -inputs[1]
    else
        symmetric = 0
    end
    for layer = 1:numLayers-1
        temp = max(*(weights[layer],inputs[1:nnet.layerSizes[layer]])+biases[layer],0)
        inputs = temp
    end
    outputs = *(weights[end],inputs[1:nnet.layerSizes[end-1]])+biases[end]
    for i=1:outputSize
        outputs[i] = outputs[i]*nnet.ranges[end]+nnet.means[end]
    end
    return outputs
end

#Evaluates multiple inputs at once. Each set of inputs should be a column in the input array
#Returns a column of output Q values for each input set
function evaluate_network_multiple(nnet::NNet,input::Array{Float64,2})
    numLayers = nnet.numLayers
    inputSize = nnet.inputSize
    outputSize = nnet.outputSize
    symmetric = nnet.symmetric
    biases = nnet.biases
    weights = nnet.weights
        
    _,numInputs = size(input)
    symmetryVec = zeros(numInputs)
    
    inputs = zeros(inputSize,numInputs)
    for i = 1:inputSize
        for j = 1:numInputs
            if input[i,j]<nnet.mins[i]
                inputs[i,j] = (nnet.mins[i]-nnet.means[i])/nnet.ranges[i]
            elseif input[i,j] > nnet.maxes[i]
                inputs[i,j] = (nnet.maxes[i]-nnet.means[i])/nnet.ranges[i] 
            else
                inputs[i,j] = (input[i,j]-nnet.means[i])/nnet.ranges[i] 
            end
            
            
            inputs[i,j] = (input[i,j]-nnet.means[i])/nnet.ranges[i] 
        end
    end
    for i=1:numInputs
        if symmetric ==1 && inputs[2,i]<0
            inputs[2,i] = -inputs[2,i]
            inputs[1,i] = -inputs[1,i]
            symmetryVec[i] = 1
        else
            symmetryVec[i] = 0
        end
    end
    
    for layer = 1:numLayers-1
        inputs = max(*(weights[layer],inputs[1:nnet.layerSizes[layer],:])+*(biases[layer],ones(1,numInputs)),0)
    end
    outputs = *(weights[end],inputs[1:nnet.layerSizes[end-1],:])+*(biases[end],ones(1,numInputs))
    for i=1:outputSize
        for j=1:numInputs
            outputs[i,j] = outputs[i,j]*nnet.ranges[end]+nnet.means[end]
        end
    end
    return outputs
end


function read_policy(actions::Matrix{Float64}, alpha::Matrix{Float64})
    return Policy(alpha, actions)
end # function read_policy

function evaluate(policy::Policy, belief::SparseMatrixCSC{Float64,Int64})
    fill!(policy.qvals, 0.0)
    get_qval!(policy, belief)
    return copy(policy.qvals)
end # function evaluate

function get_qval!(policy::Policy, belief::SparseMatrixCSC{Float64, Int64})
    fill!(policy.qvals, 0.0)
    for iaction in 1:policy.nactions
        for ib in 1:length(belief.rowval)
            policy.qvals[iaction] += belief.nzval[ib] * policy.alpha[belief.rowval[ib], iaction]
        end # for b
    end # for iaction
    #println(policy.qvals)
end # function get_qval!

function get_belief(pstate::Vector{Float64}, grid::RectangleGrid,interp::Bool=false)
    belief = spzeros(NSTATES, 1)
    indices, weights = interpolants(grid, pstate)
    if !interp
        largestWeight = 0;
        largestIndex = 0;
        for i = 1:length(weights)
            if weights[i]>largestWeight
                largestWeight = weights[i]
                largestIndex = indices[i]
            end
        end
        indices = largestIndex
        weights = largestWeight
    end
    for i = 1:length(indices)
        belief[indices[i]] = weights[i]
    end # for i
    return belief
end # function get_belief

function belief_states(r,th,psi_int,v_own,v_int,tau,pa,deltaR, deltaPsi,deltaV, deltaTheta,nnet)
    belief = zeros(9,7)
    pastActions = pas
    if nnet
        pastActions = pasTrue
    end
    belief[1,:] = [r,th,deg2rad(psi_int),v_own,v_int,tau,pastActions[pa]];
    belief[2,:] = [r+deltaR,th,deg2rad(psi_int),v_own,v_int,tau,pastActions[pa]];

    rTemp = r-deltaR
    if rTemp < 0
        rTemp = 0
    end
    belief[3,:] = [rTemp,th,deg2rad(psi_int),v_own,v_int,tau,pastActions[pa]];

    psiTemp = psi_int+deltaPsi
    if psiTemp>180.0
        psiTemp-=360.0
    end
    belief[4,:] = [r,th,deg2rad(psiTemp),v_own,v_int,tau,pastActions[pa]];

    psiTemp = psi_int-deltaPsi
    if psiTemp<-180.0
        psiTemp+=360.0
    end
    belief[5,:] = [r,th,deg2rad(psiTemp),v_own,v_int,tau,pastActions[pa]];
    belief[6,:] = [r,th,deg2rad(psi_int),v_own,v_int+deltaV,tau,pastActions[pa]];

    vTemp = v_int-deltaV
    if vTemp < 0
        vTemp = 0
    end
    belief[7,:] = [r,th,deg2rad(psi_int),v_own,vTemp,tau,pastActions[pa]];

    thTemp = th+deltaTheta
    if thTemp>pi
        thTemp-=2*pi
    end
    belief[8,:] = [r,thTemp,deg2rad(psi_int),v_own,v_int,tau,pastActions[pa]];

    thTemp = th-deltaTheta
    if thTemp<-pi
        thTemp+=2*pi
    end
    belief[9,:] = [r,thTemp,deg2rad(psi_int),v_own,v_int,tau,pastActions[pa]];
    
    return belief
    
end #function belief_states





function viz_policy(alpha::Matrix{Float64}, neuralNetworkPath::AbstractString, batch_size=500)
    
    
    nnet = NNet(neuralNetworkPath);
    grid  = RectangleGrid(pas,taus,sis,sos,psis,thetas,ranges) 
    grid2 = RectangleGrid(thetas,ranges)
    
    pstart = round(rad2deg(psis[1]),0)
    pend   = round(rad2deg(psis[end]),0)
    pdiv   = round(rad2deg(psis[2]-psis[1]),0)
    
    v0start = sos[1]
    v0end   = sos[end]
    v0div   = sos[2]-sos[1]
    
    v1start = sis[1]
    v1end   = sis[end]
    v1div   = sis[2] - sis[1]

    pastart = pas[1]
    paend   = pas[end]
    padiv   = pas[2]-pas[1]
    


    policy = read_policy(ACTIONS, alpha)
    c = RGB{U8}(1.,1.,1.) # white
    e = RGB{U8}(.0,.0,.5) # pink
    a = RGB{U8}(.0,.600,.0) # green
    d = RGB{U8}(.5,.5,.5) # grey
    b = RGB{U8}(.7,.9,.0) # neon green
    colors =[a; b; c; d; e]
    
    
    @manipulate for psi_int  = convert(Array{Int32,1},round(rad2deg(psis))),#pstart:pdiv:pend,
        v_own = sos,
        v_int = sis,
        tau in taus,
        pa = pas,
        zoom = [4, 3, 2, 1],
        nbin = [100,150,200,250],
        Interp = [false,true],
        Belief = [false,true],
        beliefProb = [0.333,0.111,0.01],
        
        deltaR   = [40, 400, 4000, 0],
        deltaTh = [5.0, 30, 60, 0],
        deltaPsi = [5, 30, 60,90, 0],
        deltaV   = [10, 100, 200, 0],
        worst    = [false,true]
        
            
            
        deltaTheta = deltaTh*pi/180.0
        
        #Load table with the inputs needed to plot the heat map
        if Belief
            numBelief = 9
            inputsNet= zeros(nbin*nbin*numBelief,7)    
            ind = 1
            for i=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                for j=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                    r = sqrt(i^2+j^2)
                    th = atan2(j,i)
                    bel = belief_states(r,th,psi_int,v_own,v_int,tau,pa,deltaR,deltaPsi,deltaV,deltaTheta,true)
                    inputsNet[ind:ind+8,:] = bel
                    ind = ind+numBelief
                end
            end
        else
            numBelief = 1
            inputsNet= zeros(nbin*nbin,7)    
            ind = 1
            for i=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                for j=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                    r = sqrt(i^2+j^2)
                    th = atan2(j,i)
                    inputsNet[ind,:] = [r,th,deg2rad(psi_int),v_own,v_int,tau,pasTrue[pa]];
                    ind = ind+1
                end
            end
        end
        
        #Calculate all of the Q values from the input array
        q_nnet = zeros(nbin*nbin*numBelief,5);
        ind = 1
        
        while ind+batch_size<nbin*nbin*numBelief
            input = inputsNet[ind:(ind+batch_size-1),:]'
            output = evaluate_network_multiple(nnet,input) 
            q_nnet = [q_nnet[1:(ind-1),:];output';q_nnet[ind+batch_size:end,:]]
            ind=ind+batch_size
        end
        input = inputsNet[ind:end,:]'
        output = evaluate_network_multiple(nnet,input)
        q_nnet = [q_nnet[1:(ind-1),:];output']

        
        ind = 1
        # Q Table Heat Map
        function get_heat1(x::Float64, y::Float64)
            r = sqrt(x^2+y^2)
            th = atan2(y,x)
            bel = belief_states(r,th,psi_int,v_own,v_int,tau,pa,deltaR,deltaPsi,deltaV,deltaTheta,false)
            qvals = evaluate(policy, get_belief(bel[1,end:-1:1][:],grid,Interp))
            if Belief
                if !worst
                    qvals*=beliefProb
                end
                for i=2:9
                    temp = evaluate(policy, get_belief(bel[i,end:-1:1][:],grid,Interp))
                    if worst
                        if minimum(temp)>minimum(qvals)
                            qvals = temp
                        end
                    else
                        qvals += temp*(1.0-beliefProb)/(numBelief-1.0)

                    end
                end
            end
            return rad2deg(ACTIONS[indmin(qvals)])
       end # function get_heat1
        
        
        #Neural Net Heat Map
       function get_heat2(x::Float64, y::Float64)              
           r = sqrt(x^2+y^2)
           th = atan2(y,x)            
           qvals = q_nnet[ind,:]
            if !worst
                qvals*=beliefProb
            end
           if Belief
               for i = 1:8
                    qvalTemp = q_nnet[ind+i,:]
                    if worst
                        if minimum(qvalTemp)>minimum(qvals)
                            qvals = qvalTemp
                        end
                    else
                        qvals+=qvalTemp*(1.0-beliefProb)/(numBelief-1.0)
                    end
               end
           end

           ind +=numBelief
           return rad2deg(ACTIONS[indmin(qvals)])
       end # function get_heat2
        
        if Belief
            g = GroupPlot(2, 2, groupStyle = "horizontal sep=3cm, vertical sep = 3cm")
            Belief = false
            push!(g, Axis([
                Plots.Image(get_heat1, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                colormap = ColorMaps.RGBArray(colors), colorbar=false),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Nominal Q Table action"))
        
            
           push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Nominal Neural Net action"))
            
            
            Belief = true
            ind = 1
            push!(g, Axis([
                Plots.Image(get_heat1, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                colormap = ColorMaps.RGBArray(colors), colorbar=false),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Belief Q Table action"))

            push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Belief Neural Net action"))
        
        else
            g = GroupPlot(2, 1, groupStyle = "horizontal sep=3cm")
            push!(g, Axis([
                Plots.Image(get_heat1, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                colormap = ColorMaps.RGBArray(colors), colorbar=false),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Q Table action"))
            
            push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Neural Net action"))
        end
        g
    end # for p_int, v0, v1, pa, ta
end # function viz_pairwise_policy




function viz_policy(neuralNetworkPath::AbstractString, batch_size=500)
    
    nnet = NNet(neuralNetworkPath);
    pstart = round(rad2deg(psis[1]),0)
    pend   = round(rad2deg(psis[end]),0)
    pdiv   = round(rad2deg(psis[2]-psis[1]),0)
    
    v0start = sos[1]
    v0end   = sos[end]
    v0div   = sos[2]-sos[1]
    
    v1start = sis[1]
    v1end   = sis[end]
    v1div   = sis[2] - sis[1]

    pastart = pas[1]
    paend   = pas[end]
    padiv   = pas[2]-pas[1]
    

    c = RGB{U8}(1.,1.,1.) # white
    e = RGB{U8}(.0,.0,.5) # pink
    a = RGB{U8}(.0,.600,.0) # green
    d = RGB{U8}(.5,.5,.5) # grey
    b = RGB{U8}(.7,.9,.0) # neon green
    colors =[a; b; c; d; e]
    
    
    @manipulate for psi_int  = convert(Array{Int32,1},round(rad2deg(psis))),#pstart:pdiv:pend,
        v_own = sos,
        v_int = sis,
        tau in taus,
        pa = pas,
        zoom = [4, 3, 2, 1],
        nbin = [100,150,200,250],
        Belief = [false,true],
        beliefProb = [0.333,0.111,0.01],
        
        deltaR   = [40, 400, 4000, 0],
        deltaTh  = [5.0, 30, 60, 0],
        deltaPsi = [5, 30, 60,90, 0],
        deltaV   = [10, 100, 200, 0],
        worst    = [false,true]
        
            
            
        deltaTheta = deltaTh*pi/180.0
        
        #Load table with the inputs needed to plot the heat map
        if Belief
            numBelief = 9
            inputsNet= zeros(nbin*nbin*numBelief,7)    
            ind = 1
            for i=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                for j=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                    r = sqrt(i^2+j^2)
                    th = atan2(j,i)
                    bel = belief_states(r,th,psi_int,v_own,v_int,tau,pa,deltaR,deltaPsi,deltaV,deltaTheta,true)
                    inputsNet[ind:ind+8,:] = bel
                    ind = ind+numBelief
                end
            end
        else
            numBelief = 1
            inputsNet= zeros(nbin*nbin,7)    
            ind = 1
            for i=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                for j=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                    r = sqrt(i^2+j^2)
                    th = atan2(j,i)
                    inputsNet[ind,:] = [r,th,deg2rad(psi_int),v_own,v_int,tau,pasTrue[pa]];
                    ind = ind+1
                end
            end
        end
        
        #Calculate all of the Q values from the input array
        q_nnet = zeros(nbin*nbin*numBelief,5);
        ind = 1
        
        while ind+batch_size<nbin*nbin*numBelief
            input = inputsNet[ind:(ind+batch_size-1),:]'
            output = evaluate_network_multiple(nnet,input) 
            q_nnet = [q_nnet[1:(ind-1),:];output';q_nnet[ind+batch_size:end,:]]
            ind=ind+batch_size
        end
        input = inputsNet[ind:end,:]'
        output = evaluate_network_multiple(nnet,input)
        q_nnet = [q_nnet[1:(ind-1),:];output']

        
        ind = 1       
        #Neural Net Heat Map
       function get_heat2(x::Float64, y::Float64)              
           r = sqrt(x^2+y^2)
           th = atan2(y,x)            
           qvals = q_nnet[ind,:]
            if !worst
                qvals*=beliefProb
            end
           if Belief
               for i = 1:8
                    qvalTemp = q_nnet[ind+i,:]
                    if worst
                        if minimum(qvalTemp)>minimum(qvals)
                            qvals = qvalTemp
                        end
                    else
                        qvals+=qvalTemp*(1.0-beliefProb)/(numBelief-1.0)
                    end
               end
           end

           ind +=numBelief
           return rad2deg(ACTIONS[indmin(qvals)])
       end # function get_heat2
        
        if Belief
            g = GroupPlot(2, 1, groupStyle = "horizontal sep=3cm, vertical sep = 3cm")
            Belief = false
            
           push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Nominal Neural Net action"))
            
            
            Belief = true
            ind = 1
            push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                    ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Belief Neural Net action"))
        
        else
            g = GroupPlot(1, 1, groupStyle = "horizontal sep=3cm")
            push!(g, Axis([
               Plots.Image(get_heat2, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                           zmin = -3, zmax = 3,
                           xbins = nbin, ybins = nbin,
                           colormap = ColorMaps.RGBArray(colors)),
               Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
               Plots.Node(L">", 53000/zoom, 53000/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
                ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Neural Net action"))
        end
        g
    end # for p_int, v0, v1, pa, ta, etc
end # function viz_pairwise_policy

end #module PolicyViz