%create a synthetic image with single textured region
%Put texels at random location

clear all;
close all;

imageWidth = 400;
imageHeight = 400;
texelwidth = 10;
texelheight = 10;
orientation1degrees = 30;
orientation1rad =   orientation1degrees * (pi/180);

im1 = ones(imageWidth,imageHeight);
numberOfTexels = 1000;
numberOfpointsAdded = 0;
while(numberOfpointsAdded < numberOfTexels)
    x = randi(imageWidth);
    y = randi(imageHeight);
    
    %check if (x,y) is near boundary
    if(x - texelwidth/2  + 1 > 1 && x + texelwidth/2 < imageWidth && y - texelheight/2 + 1 > 1 && y + texelheight/2 <imageHeight)
        %check for overlapping texels
        window = im1(x - texelwidth/2 + 1 :x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 );
        if(min(min(window)) ==1)
            texel = createTexel1(texelheight,texelwidth,orientation1rad ,1,0);
            im1(x - texelwidth/2 + 1:x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 ) = texel;
            numberOfpointsAdded = numberOfpointsAdded + 1;
        end
        
    end
    
    
end

orientation2degrees =  10;
orientation2rad = orientation2degrees * (pi/180);

im2 = ones(imageWidth,imageHeight);
numberOfTexels = 1000;
numberOfpointsAdded = 0;
while(numberOfpointsAdded < numberOfTexels)
    x = randi(imageWidth);
    y = randi(imageHeight);
    
    %check if (x,y) is near boundary
    if(x - texelwidth/2  + 1 > 1 && x + texelwidth/2 < imageWidth && y - texelheight/2 + 1 > 1 && y + texelheight/2 <imageHeight)
        %check for overlapping texels
        window = im2(x - texelwidth/2 + 1 :x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 );
        if(min(min(window)) ==1)
            texel = createTexel1(texelheight,texelwidth,orientation2rad,1,0);
            im2(x - texelwidth/2 + 1:x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 ) = texel;
            numberOfpointsAdded = numberOfpointsAdded + 1;
        end
        
    end
    
    
end

im = [im1;im2];

fileName =strcat('C:\TextureDL\data\RandomSpacing',int2str(orientation1degrees),'And',int2str(orientation2degrees),'DegreeLine',int2str(numberOfTexels),'Texel.jpg');
fileName1 =strcat('C:\TextureDL\data\RandomSpacing',int2str(orientation1degrees),'DegreeLine',int2str(numberOfTexels),'Texel.jpg');
fileName2 =strcat('C:\TextureDL\data\RandomSpacing',int2str(orientation2degrees),'DegreeLine',int2str(numberOfTexels),'Texel.jpg');
imwrite(im,fileName,'jpg');

imwrite(im1,fileName1,'jpg');
imwrite(im2,fileName2,'jpg');

imshow(im);
figure;
imshow(im1);
figure;
imshow(im2);

