%Author : Sunil Kumar Vengalil
%create a synthetic image with single textured region
%randomize veritcal and horizontal spacing
clear all
close all
fileName ='C:\TextureDL\Data\RandomSpacing5.jpg';
%ImageWidth and ImageHeight are in the units of number of Texels
imageWidth = 400;
imageHeight = 10;

linelength = 5;
maxHeight = 12;
maxWidth =12;
orientation = 0;


%create texels
linetexel = createTexel1(maxHeight,maxWidth,orientation,1,0);
spacingVariation = 10;

line1 = createRandomLine(linetexel,spacingVariation,imageWidth,25);


im = line1;

for x = 1: imageHeight
    im = [im;createRandomLine(linetexel,spacingVariation,imageWidth,45)];
end



orientation = 3 * pi /4;

linetexel = createTexel1(maxHeight,maxWidth,orientation,1,0);
spacingVariation = 10;

line1 = createRandomLine(linetexel,spacingVariation,imageWidth,25);


im1 = line1;

for x = 1: imageHeight
    im1 = [im1;createRandomLine(linetexel,spacingVariation,imageWidth,45)];
end

im = [im1;im]



imwrite(im,fileName,'jpg');
imshow(im);
