%create images with four different oriented linesegments
%TODO color
clc;
close all;
clear all;

imfolderName = 'im';
labelfolderName = 'label';
imageWidth = 200;
imageHeight = 400;


im45 = createDataset(135,imageHeight,imageWidth);
imv = createDataset(0,imageHeight,imageWidth);
imh = createDataset(90,imageHeight,imageWidth);
im135 = createDataset(45,imageHeight,imageWidth);

index = 10000;
numberOfImage = 1200;
numRegionsPerImage = 3;
x =  rand(numberOfImage * numRegionsPerImage,1);
x1 =  round(rand(numberOfImage,1));

%label1 = [ones(200,200);zeros(200,200)];
%label2 = [zeros(200,200);ones(200,200)];
%label3 = [zeros(200,200);(200,200)];

rootDir = 'C:\TextureDL\data5\';
vl_xmkdir(fullfile(rootDir));
vl_xmkdir(fullfile(rootDir,imfolderName));
vl_xmkdir(fullfile(rootDir,labelfolderName));


for i = 1:numRegionsPerImage:numberOfImage * numRegionsPerImage
    index1 = randi(50);
    %index2 = randi(50);
    fileName =strcat(rootDir,imfolderName,'\im', int2str(index),'.jpg');
    labelFileName = strcat(rootDir,labelfolderName,'\im', int2str(index),'.jpg');
    p = x(i);
    [image,label]=chooseImageAndLabel(p,index1,imv,imh,im45,im135 );
    
    p = x(i+1);
    [image1,label1]=chooseImageAndLabel(p,index1,imv,imh,im45,im135 );
    
            
    for row = 1:200
        for col = row : 200
            image(row,col) = image1(row,col);
            label(row,col) = label1(row,col);
        end
    end
    
    p = x(i+2);
    [image1,label1]=chooseImageAndLabel(p,index1,imv,imh,im45,im135 );
    
    for row = 200:400
        for col = 400-row+1 : imageWidth
            image(row,col) = image1(row,col);
            label(row,col) = label1(row,col);
        end
     end
        
    imwrite(image,fileName,'jpg');
    imwrite(uint8(label),labelFileName,'jpg');
    index  = index + 1;
        
end
        

