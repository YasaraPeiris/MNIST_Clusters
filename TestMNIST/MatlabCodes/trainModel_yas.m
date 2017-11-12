
function trainModel_yas(layerset, dataSize) % Using Oja's rule

trainingRatio = 0.8;
p = 0;

images = loadTrainImages();
labels = loadTrainLabels();


selected = find(labels == 2 | labels == 1 );
labels = labels(selected);
images = images(:, selected');

[~, c] = size(images);

dataSize = min(c, dataSize);
iterations = dataSize;

image_batch = 1;

testLabels = [];
clusters = [];

trainingSize = floor(double(dataSize) * trainingRatio)/image_batch;

unclassified = 0;
norms = [];

updateTime = 0.0;
testImageStartId = iterations;
test_image = [];
test_label = [];

for i =1:image_batch
    test_image  = [test_image mat2gray(images(:,testImageStartId+i ))];
    test_label = [test_label labels(testImageStartId+i)];
    
end
%{
im = vec2mat(images(:, randi(10000)), 28)';
imshow(im);
drawnow;
%}

%showFinalImage(weights{1});
%temp = weights;

net = Network_new([784, layerset, 2]);
numLayers = net.numLayers;
tempW = net.feedforwardConnections;
%tempW = net.lateralConnections;
temp = net.feedforwardConnections;
image_count = 0;
test_count = 0;
pow = 1;
count_labels_1=0;
count_labels_2=0;
for r= 1:iterations/image_batch
    images_new = [];
    for k=1:image_batch
        
        image_id = image_batch*(r-1)+k;
        images_new = [images_new mat2gray(images(:, image_id))];
     
        
    end
    
    results = net.getOutput(images_new,r);
    if(r < trainingSize)
%         dlmwrite('max_4.txt','train ---------------------------','-append');
        for u=1:image_batch
            image_id = image_batch*(r-1)+u;
            [m, ~] = max(results{numLayers}(:,u));
             
%             dlmwrite('max_4.txt','max','-append');
%             dlmwrite('max_4.txt',results{numLayers}(:,u),'-append');
%              dlmwrite('max_4.txt','i','-append');
%             dlmwrite('max_4.txt',i,'-append');
%             dlmwrite('max_4.txt','label','-append');
%             dlmwrite('max_4.txt',labels(image_id),'-append');
        end
    end
    
    time = tic;
    net.STDP_update(results,r);
    updateTime = updateTime + toc(time);
    for u = 1 : image_batch
    
    norms = [norms; zeros(1, numLayers - 1)];
    
    weights = net.feedforwardConnections;
    %     weights = net.lateralConnections;
    
    for k = 1 : numLayers - 1
        
        norms(end, k) = norm(weights{k}(:,u) - tempW{k}(:,u),'fro') / numel(weights{k}(:,u));
        tempW{k}(:,u) = weights{k}(:,u);
        %         disp(norms);
    end
    
    end
end

for h = 1: margin/image_batch
    test_image_batch=[];
    for k=1:image_batch
        
        image_id = image_batch*(h-1)+k;

        test_image_batch = [test_image_batch test_image(:,image_id)];
        
        
    end
    
    
    results = net.getOutput(test_image_batch,newIterations+1,-1);
    for u=1:image_batch
        
        %     columns = ['A1','B1','C1','D1','E1','F1','G1','H1','I1','J1'];
        %     xlswrite('weight_15.xlsx',results{1});
        %     xlswrite('weight_16.xlsx',results{4});
        
        [m, i] = max(results{numLayers}(:,u));
        m
        if(m >= p)
            %                 image_id = image_batch*(r-1)+u;
            testLabels = [testLabels; test_label(:,u)];
            %               testLabels = [testLabels; labels(r)];
            clusters = [clusters; i];
           
        else
            %                 disp(unclassified);
            unclassified = unclassified + 1;
            
        end
    end
end

plotPerformance([1 : iterations]', norms, testLabels, clusters, [1, 2, 3]);

% disp(['Unclassified: ', int2str(unclassified), ' out of ', int2str(dataSize - trainingSize)]);
%
% disp(['Average STDP update time = ', num2str(updateTime / iterations/image_batch)]);

for r = 1 : numLayers - 1
    
    disp([int2str(r),': ', int2str(net.ffcheck(r))]);
    sheet=1;
%     xlswrite('weight_2.xlsx','new layer',sheet);
%     xlswrite('weight_2.xlsx',weights{r},sheet);
    
end

%{
for i = 1 : numLayers - 1
    
    showFinalImage(weights{i});
   
end
%}

%showFinalImage([temp{1}, max(max(weights{1}))* ones(layers(2), 5), weights{1}]);

%showFinalImage(weights{1});

showFinalImage(abs(weights{1} - temp{1}));

%clust = kmeans(images(:, trainingSize + 1 : dataSize)', 8);

%plotPerformance([1 : iterations]', norms, testLabels, clust, [2, 3]);

%disp(clusters);




