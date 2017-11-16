
function trainModel_yas(layerset, dataSize) % Using Oja's rule

trainingRatio = 0.8;
p = 0;

images = loadTrainImages();
labels = loadTrainLabels();


selected = find(labels == 0 | labels == 1 );
labels = labels(selected);
images = images(:, selected');

[~, c] = size(images);

dataSize = min(c, dataSize);
iterations = dataSize;

image_batch = 10;

newIterations = fix(iterations/image_batch);
testLabels = [];
clusters = [];

trainingSize = floor(double(dataSize) * trainingRatio)/image_batch;

unclassified = 0;
norms = [];

updateTime = 0.0;

net = Network_new([784, layerset, 2]);
numLayers = net.numLayers;
tempW = net.feedforwardConnections;
%tempW = net.lateralConnections;
temp = net.feedforwardConnections;

for r= 1:newIterations
    images_new = [];
    for k=1:image_batch
        
        image_id = image_batch*(r-1)+k;
        images_new = [images_new mat2gray(images(:, image_id))];
           
    end
    
    results = net.getOutput(images_new,r);
    
    if(r > trainingSize)
       
            for u=1:image_batch
                
                image_id = image_batch*(r-1)+u;
                [m, i] = max(results{numLayers}(:,u));
                testLabels = [testLabels; labels(image_id)];
                clusters = [clusters; i];
                
            end
           
    end
    
    time = tic;
    net.STDP_update(results,r);
    updateTime = updateTime + toc(time);
    for u = 1 : image_batch
        
        norms = [norms; zeros(1, numLayers - 1)];
        
        weights = net.feedforwardConnections;
        
        for k = 1 : numLayers - 1
            
            norms(end, k) = norm(weights{k}(:,u) - tempW{k}(:,u),'fro') / numel(weights{k}(:,u));
            tempW{k}(:,u) = weights{k}(:,u);
            
        end
        
    end
end


plotPerformance([1 : iterations]', norms, testLabels, clusters, [1, 2, 3]);

showFinalImage(abs(weights{1} - temp{1}));





