module PolicyViz_DRL

export viz_policy_drl
using Interact, PGFPlots, Colors, ColorBrewer



const RANGEMAX = 3000.0 # meters
const RANGEMIN = 0.0 # meters
const PsiDim   = 41
const PsiMin   = -pi #[rad]
const PsiMax   = pi  #[rad]

const psis      = linspace(PsiMin,PsiMax,PsiDim)
const sos       = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
const sis       = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

const Actions= [-20.0,-10.0,0.0,10.0,20.0,-6.0]

const STATE_DIM = 5
const ACTION_DIM = 6     


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


function viz_policy_drl(neuralNetworkPath::AbstractString, batch_size=500)
    
    println("here")
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
    
    c = RGB{U8}(1.,1.,1.) # white
    e = RGB{U8}(.0,.0,.5) # pink
    a = RGB{U8}(.0,.600,.0) # green
    d = RGB{U8}(.5,.5,.5) # grey
    b = RGB{U8}(.7,.9,.0) # neon green
    f = RGB{U8}(0.94,1.0,.7) # pale yellow
    colors =[a; b; f; c; d; e]
   
    @manipulate for psi_int  = round(rad2deg(psis)),
        v_own = sos,
        v_int = sis,
        zoom = [4, 3, 2, 1.5,1],
        nbin = [100,150,200,250]
            
        #mat =  ccall((:load_network,LIB_BLAS),Ptr{Void},(Ptr{UInt8},),neuralNetworkPath)
        inputsNet= zeros(nbin*nbin,STATE_DIM)    
        ind = 1
        for i=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
            for j=linspace(round(Int,-1*RANGEMAX/zoom),round(Int,RANGEMAX/zoom),nbin)
                r = sqrt(i^2+j^2)
                th = atan2(j,i)
                inputsNet[ind,:] = [r,th,deg2rad(psi_int),v_own,v_int];
                ind = ind+1
            end
            end            

        q_nnet = zeros(nbin*nbin,ACTION_DIM);
        ind = 1
        while ind+batch_size<nbin*nbin            
            input = inputsNet[ind:(ind+batch_size-1),:]'
            output = evaluate_network_multiple(nnet,input) 
            q_nnet = [q_nnet[1:(ind-1),:];output';q_nnet[ind+batch_size:end,:]]
            ind=ind+batch_size
        end
        input = inputsNet[ind:end,:]'
        output = evaluate_network_multiple(nnet,input)
        q_nnet = [q_nnet[1:(ind-1),:];output']
        
        ind = 1       
        function get_heat(x::Float64, y::Float64)              
           r = sqrt(x^2+y^2)
           th = atan2(y,x)            
            action  = Actions[indmax(q_nnet[ind,:])]
            ind = ind+1
            return action
        end # function get_heat2
        
        g = GroupPlot(1, 1, groupStyle = "horizontal sep=3cm")
        push!(g, Axis([
           Plots.Image(get_heat, (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                       (round(Int,-1*RANGEMAX/zoom), round(Int,RANGEMAX/zoom)), 
                       zmin = -20, zmax = 20,
                       xbins = nbin, ybins = nbin,
                       colormap = ColorMaps.RGBArray(colors)),
           Plots.Node(L">", 0, 0, style=string("font=\\Huge")),
           Plots.Node(L">", 2500/zoom, 2500/zoom, style=string("rotate=", psi_int, ",font=\\Huge"))
            ], width="10cm", height="10cm", xlabel="x (ft)", ylabel="y (ft)", title="Neural Net action"))
        g
    end # for p_int, v0, v1, pa, ta
end # function viz_policy_drl

end #module PilotSCAViz
