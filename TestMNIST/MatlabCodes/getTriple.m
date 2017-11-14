function result = getTriple(images)
[~,n]=size(images);
images_final = [];
for k = 1:n

temp = images(:,k);
count = 0;
for i =1:784
    images_new = [];
    for j = 1: 3
    images_new = [images_new; temp ];
    end
end


images_final = [images_final, images_new];
end
result = images_final;