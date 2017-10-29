classdef Network_new < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (GetAccess = private)
    
        weightFile;
        
        t = 0.001;
        a = 15.0;
        b = [0.01, 0.01, 0.025];
        %b = [0.0, 0.0, 0.0];
        
    end
    
    properties (SetAccess = public)
        
        layerStruct;
        numLayers;
        totalRounds;
        ffcheck;
        ltcheck;
        fbcheck;
        
        feedforwardConnections;
        lateralConnections;
        feedbackConnections;
        
    end
    
    properties
        
        
    end
    
    methods (Access = private)
        
        function createFeedforward(obj)
            
            if exist(obj.weightFile, 'file') == 2
                load(obj.weightFile, 'feedforwardConnections');
                obj.feedforwardConnections = feedforwardConnections;
            else
                            
                obj.feedforwardConnections = cell([1, obj.numLayers - 1]);

                for i = 1 : obj.numLayers - 1

                    %obj.feedforwardConnections{i} = rand(layerStruct(i + 1),layerStruct(i));
                    %obj.feedforwardConnections{i} = normr(binornd(1, 0.2, obj.layerStruct(i + 1), obj.layerStruct(i)));
                    obj.feedforwardConnections{i} = binornd(1, 0.2, obj.layerStruct(i + 1), obj.layerStruct(i));

                end
                
            end
            
            obj.ffcheck = zeros(1, obj.numLayers - 1);
            
        end
        
        function createLateral(obj)
            
%             if exist(obj.weightFile, 'file') == 2
%                 load(obj.weightFile, 'lateralConnections');
%                 obj.lateralConnections = lateralConnections;
%             else
            
                obj.lateralConnections = cell([1, obj.numLayers - 1]);

                for i = 1 : obj.numLayers - 1

                    %obj.lateralConnections{i} = rand(layerStruct(i + 1),layerStruct(i + 1));
                    obj.lateralConnections{i} = - normr(binornd(1, 0.2, obj.layerStruct(i + 1), obj.layerStruct(i + 1)));

                    obj.lateralConnections{i}(1 : obj.layerStruct(i + 1) + 1 : obj.layerStruct(i + 1) * obj.layerStruct(i + 1)) = 0;
for r = 1 : obj.layerStruct(i + 1)
    for k = 1 : obj.layerStruct(i + 1)
        if r~=k-1 || r~= k+1
        obj.lateralConnections{i}(r,k) = 0;
        end
    end
end
    
                end
                
%             end
            
            obj.ltcheck = zeros(1, obj.numLayers - 1);
            
        end
        
        function createFeedback(obj,r,temp)
            
            this_a = obj.a;
            temp(temp < 0) = 0;
            temp = temp - 0.005;
            
            
            obj.feedforwardConnections{r} = this_a * temp + obj.feedforwardConnections{r};
           
        end
        
        function STDP_update_feedforward_old(obj, layers)
            parfor r = 1 : obj.numLayers - 1
                mean_A = mean(layers{r}');
                mean_B = mean(layers{r+1}');
                [m,n] = size(layers{r});
                total_product = [];
                for k = 1 :n
                    total_temp  = layers{r+1}(k) * (layers{r}(k))';
                    total_product = total_product + layers{r+1}(k) * (layers{r}(k))';
                end
                total_product = total_product./10;
                temp = total_product - (mean_A')*(mean_B);
                weights{r} = weights{r} + this_t * temp;
                
%                 obj.createFeedback(r,temp);
               
                if any(temp < 0)
                    this_check(r) = this_check(r) + 1;
                end
            end
            obj.feedforwardConnections = weights;
            obj.ffcheck = this_check;
        end
        
        function saveWeights(obj)
            
            feedforwardConnections = obj.feedforwardConnections;
%              lateralConnections = obj.lateralConnections;
%             save(obj.weightFile, 'feedforwardConnections', 'lateralConnections');
             save(obj.weightFile, 'feedforwardConnections');
            
        end
        
    end
    
    methods
        
        function obj = Network_new(layerStruct)
            
            obj.layerStruct = layerStruct;
            [~, obj.numLayers] = size(layerStruct);
            obj.totalRounds = 0;          
            
            fileName = sprintf('%d_', layerStruct);
            fileName = strcat(fileName(1 : end - 1), '.mat');
            obj.weightFile = fullfile(fileparts(which(mfilename)), '..\WeightDatabase\Temp', fileName);
            
            obj.createFeedforward();
%              obj.createLateral();
         
            obj.saveWeights();
            
        end
        
        function layers = getOutput(obj, input)
            [~,n]=size(input);
            layers = cell([1, obj.numLayers]);
            layers{1} = normc(input);
            layers_batch = zeros(784, n);
            for k = 1 : obj.numLayers - 1
                for g = 1 : n
                    if k == 1
                        
                        layers_batch(:,g) = obj.feedforwardConnections{k}(:,g)* layers{1}(:,g);
                        
                    else
                        
                        layers_batch(:,g) = obj.feedforwardConnections{k}(:,g)* layers_batch(:,g);
                    
                    end
                    
                end
              
                    layers{k + 1} = layers_batch;
                    layers{k + 1} = normc(layers{k + 1});             
            end
            
        end
        
        function STDP_update(obj, layers)
                       
            obj.totalRounds = obj.totalRounds + 1;
            obj.STDP_update_feedforward(layers); 
%          obj.STDP_update_lateral(layers);
        
        end
        
    end
    
end

