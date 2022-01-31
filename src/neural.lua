-- Credits:
-- net made by ChickenSauceSandwich, discord: Bald man with no hair#8606
-- edited by GForcebot, discord: G Kitteh Cat#7884
-- cleaned up by me, tech, discord: n/a
local neuralNetwork = {}
local function bool2int(bool)
	if (bool == true) then
		return 1
	else
		return 0
	end
end
local learningRate = .15
-- sigmoid activation local function
local function sigmoid(activation)
	return 1.0 / (1.0 + math.exp(-activation))
end

local function initNeuron(inputs)
	local neuron = {weights = {}, inputs = {}, activation = 0, delta = 0}

	-- initialize input weights
	for _ = 1, inputs do
		table.insert(neuron.weights, math.random() * (1 - (-1)) - 1)
	end

	-- initialize bias weight
	table.insert(neuron.weights, 0)

	return neuron
end

local function getActivation(neuron, activationf)
	local activation = neuron.weights[#neuron.weights]

	for w = 1, #neuron.weights - 1 do
		activation += neuron.inputs[w] * neuron.weights[w]
	end

	if (activationf == "sigmoid") then
		activation = sigmoid(activation)
	elseif (activationf == "tanh") then
		activation = math.tanh(activation)
	elseif (activationf == "ReLU") then
		activation = math.max(activation * 0.01, activation)
	end

	return activation
end

-- get slope of activation
local function transferDerivative(activationf, activation)
	if (activationf == "sigmoid") then
		activation = sigmoid(activation) * (1 - sigmoid(activation))
	elseif (activationf == "tanh") then
		activation = 1 - math.tanh(activation)^2
	elseif (activationf == "ReLU") then
		activation = 1 * bool2int((activation > 0))
	end

	return activation
end

-- calculates network output
local function propogateForward(input)
	local oldInput = {}
	for l, layer in pairs(neuralNetwork) do
		local newInput = {}
		for n, neuron in pairs(layer) do
			if (#neuron.inputs > 0) then
				neuron.inputs = {}
			end

			if (l  ==  1) then
				table.insert(neuron.inputs, input[n])
			else
				neuron.inputs = oldInput
			end

			-- get activation
			local activation = getActivation(neuron, "sigmoid")
			neuron.activation = activation
			table.insert(newInput, activation)
		end
		oldInput = newInput
		newInput = {}
	end
	return oldInput
end
-- calculates the error of the network
local function propogateBackwards(label)
	local lastLayer = {}
	for l = #neuralNetwork, 1, -1 do
		local losses = {}
		if (l ~= #neuralNetwork) then
			for n, _ in pairs(neuralNetwork[l]) do
				-- calculate loss of hidden neurons
				local loss = 0
				for _, pneuron in pairs(lastLayer) do
					loss += pneuron.weights[n] * pneuron.delta
				end
				table.insert(losses,loss)
			end
		else
			for n, neuron in pairs(neuralNetwork[l]) do
				table.insert(losses,neuron.activation - label[n])
			end
		end

		for n, neuron in pairs(neuralNetwork[l]) do
			neuron.delta = losses[n] * transferDerivative("sigmoid", neuron.activation)
		end

		losses = {}
		lastLayer = neuralNetwork[l]
	end
end

-- local function to update neuron weights
local function updateNetwork()
	for _, layer in pairs(neuralNetwork) do
		for _, neuron in pairs(layer) do
			for i, _ in pairs(neuron.inputs) do
				neuron.weights[i] -= learningRate * neuron.delta * neuron.inputs[i]
			end

			-- update bias weight
			neuron.weights[#neuron.weights] -= learningRate * neuron.delta
		end
	end
end
-- input layer
local layers = {}
for _ = 1,3 do
	local neur = initNeuron(1)
	table.insert(layers,neur)
end
table.insert(neuralNetwork, layers)
-- hidden
local layers = {}
for _ = 1,7 do
	local neuron = {}
	neuron = initNeuron(3)
	table.insert(layers, neuron)
end
table.insert(neuralNetwork, layers)
local layers = {}
for i = 1, 7 do
	local neuron = {}
	neuron = initNeuron(7)
	table.insert(layers, neuron)
end
table.insert(neuralNetwork, layers)
-- output
local layers = {}
for i = 1, 3 do
	local neuron = initNeuron(7)
	table.insert(layers, neuron)
end
table.insert(neuralNetwork, layers)

return {
	propogateForward = propogateForward,
	propogateBackwards = propogateBackwards,
	updateNetwork = updateNetwork,
}