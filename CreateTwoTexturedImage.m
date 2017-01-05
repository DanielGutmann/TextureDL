%create images with two different textures
clc;
close all;
clear all;

imfolderName = 'im';
labelfolderName = 'label';
orientation1degrees = 45;
orientation1rad  =   orientation1degrees * (pi/180);
index = 1;

%TODO color

imageWidth = 200;
imageHeight = 200;
numberOfTexels = 250;
im45 = zeros(imageHeight,imageWidth,50);
for texelwidth = 8:2:12
    for texelheight = 8:2:10
        for fgGrayValue =130:10:250
            fwrite(1,sprintf('%d %d \n',texelwidth,texelheight));
            texel = createTexel2(texelheight,texelwidth,orientation1rad ,fgGrayValue,0);
            
            im45(:,:,index) = createSyntheticImageRandomSpacing(imageWidth,imageHeight,texel,numberOfTexels);
            
            index = index + 1;
        end
    end
end

imv = zeros(imageHeight,imageWidth,50);
orientation1degrees = 90;
orientation1rad  =   orientation1degrees * (pi/180);
index = 1;

for texelwidth = 8:2:10
    for texelheight = 8:2:10
        for fgGrayValue =130:10:250
            fwrite(1,sprintf('%d %d \n',texelwidth,texelheight));
            texel = createTexel2(texelheight,texelwidth,orientation1rad ,fgGrayValue,0);
            
            imv(:,:,index) = createSyntheticImageRandomSpacing(imageWidth,imageHeight,texel,numberOfTexels);
            
            index = index + 1;
        end
    end
end



index = 1;
x =  round(rand(400,1));

label1 = [ones(200,200);zeros(200,200)];
label2 = [zeros(200,200);ones(200,200)];
for i = 1:400
    index1 = randi(50);
    index2 = randi(50);
    fileName =strcat('C:\TextureDL\data\',imfolderName,'\im', int2str(index),'.jpg');
    labelFileName = strcat('C:\TextureDL\data\',labelfolderName,'\im', int2str(index),'.jpg');
    if x(i) == 1
        imwrite([imv(:,:,index1);im45(:,:,index2)],fileName,'jpg');
        imwrite(label1,labelFileName,'jpg');
    else
        imwrite([im45(:,:,index1);imv(:,:,index2)],fileName,'jpg');
        imwrite(label2,labelFileName,'jpg');
    end
    index = index + 1;
end






