classdef Network_new < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (GetAccess = private)
        
        weightFile;
        
        t = 0.001;
        a = 15.0;
        b = [0.01, 0.01, 0.025];
        %b = [0.0, 0.0, 0.0];
        count_1 =0;
        count_2=0;
        image_label = 0;
    end
    
    properties (SetAccess = public)
        
        layerStruct;
        numLayers;
        totalRounds;
        ffcheck;
        ltcheck;
        fbcheck;
        iterationImages;
       
        
        feedforwardConnections;
        lateralConnections;
        feedbackConnections;
        
    end
    
    properties
        
        
    end
    
    methods (Access = private)
        function createLateral(obj)
            
%             if exist(obj.weightFile, 'file') == 2
%                 load(obj.weightFile, 'lateralConnections');
%                 obj.lateralConnections = lateralConnections;
%             else
            
                obj.lateralConnections = cell([1, obj.numLayers - 1]);

                for i = 1 : obj.numLayers - 1

                    %obj.lateralConnections{i} = rand(layerStruct(i + 1),layerStruct(i + 1));
                    obj.lateralConnections{i} = - normr(binornd(1, 0.2, obj.layerStruct(i + 1), obj.layerStruct(i + 1)));

                    obj.lateralConnections{i}(1 : obj.layerStruct(i + 1) + 1 : obj.layerStruct(i + 1) * obj.layerStruct(i + 1)) = 1;

                end
                
%             end
            
            obj.ltcheck = zeros(1, obj.numLayers - 1);
            
        end
        
        function createFeedforward(obj)
            
%             if exist(obj.weightFile, 'file') == 2
%                 load(obj.weightFile, 'feedforwardConnections');
%                 obj.feedforwardConnections = feedforwardConnections;
%             else
                
                obj.feedforwardConnections = cell([1, obj.numLayers - 1]);
                
                for i = 1 : obj.numLayers - 1
                    
                    obj.feedforwardConnections{i} = rand(obj.layerStruct(i + 1),obj.layerStruct(i));
                end
            obj.ffcheck = zeros(1, obj.numLayers - 1);
            
        end
        
        
       
        
        function STDP_update_feedforward(obj, layers, iteration)
            weights = obj.feedforwardConnections;
            this_t = obj.t;
            this_check = obj.ffcheck;
            this_totalRounds = obj.iterationImages;
%             
            parfor r = 1 : obj.numLayers - 1
                
                mean_A = mean(layers{r}');
                mean_B = mean(layers{r+1}');
                
                [m,n] = size(layers{r});
                
                [e,l] = size((mean_A')*(mean_B));
                 total_product = zeros(l,e);

                for k = 1 : n
                    
                    total_temp  = layers{r+1}(:,k) * (layers{r}(:,k))'*0.000099;
                    total_product = total_product + total_temp;
                    
                end
                total_product = total_product./n;
                temp = total_product - 0.00095*((mean_A')*(mean_B))';
                temp = (temp*(exp(-0.008*iteration))/r);
                 
                if(iteration == this_totalRounds && r==1)
                    xlswrite('temp_1.xlsx',temp);
                    xlswrite('weight_1',weights{r}(:,1:50));
                end
                 if(iteration == this_totalRounds && r==2)
                    xlswrite('temp_2.xlsx',temp);
                    xlswrite('weight_2.xlsx',weights{r}(:,1:50));
                 end
                 if(iteration == this_totalRounds && r==3)
                    xlswrite('temp_3.xlsx',temp);
                    xlswrite('weight_3.xlsx',weights{r});
                 end

                weights{r} = weights{r} + temp;
        
                if any(temp == 0)
                    this_check(r) = this_check(r) + 1;
                end
            end
            
            obj.feedforwardConnections = weights;
            obj.ffcheck = this_check;
            
        end
        
        function saveWeights(obj)
            
            feedforwardConnections = obj.feedforwardConnections;
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
            obj.saveWeights();
            
        end
        
        function layers = getOutput(obj, input,iteration,label)
            this_totalRounds = obj.iterationImages;
                    
            [m,n]=size(input);
            
            layers = cell([1, obj.numLayers]);  
              input(input<0) = 0;
              input(input>0) = 1;

            layers{1} = input;
          
            sheet =1;
            
            for k = 1 : obj.numLayers - 1
                
             
                  layers{k + 1} = obj.feedforwardConnections{k}* layers{k};
                  layers{k + 1} = layers{k + 1}/norm(layers{k + 1},1.0);
    
            end
            
              if(iteration>this_totalRounds)

                      xlswrite('final_3_5.xlsx',layers{3});
                      xlswrite('final_4_5.xlsx',layers{4});
              end
           
        end
        
        
        function STDP_update(obj, layers, r)
            
            obj.totalRounds = obj.totalRounds + 1;
            obj.STDP_update_feedforward(layers, r);
            
        end
        
    end
    
end

