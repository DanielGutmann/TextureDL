
%create a sythetic image with single textured region
%randomize veritcal and horizontal spacing
clear all
close all


fileName ='C:\TextureDL\data\Image10.jpg';
%ImageWidth and ImageHeight are in the units of number of Texels
imageWidth = 450;
imageHeight = 20;

linelength = 5;
maxHeight = 12;
maxWidth =12;
%create texels
linetexel = createTexel1(maxHeight,maxWidth,3*(pi/4),1,0);



spacingVariation = 10;
heightVariation = maxHeight-2*linelength;
minimumWidth = 2 * linelength + 1;
minimumHeight = 2 * linelength + 1;
%repeat the texel to create row
%line1 = zeros(imageWidth,1);

texel = randomizeHeight( linetexel,spacingVariation,25);
line1 = texel;
while size(line1,2) < imageWidth -  (spacingVariation + size(texel,2))
    blank = ones(size(texel,1), randi(spacingVariation));
    line1 = [line1 blank];
    texel = randomizeHeight( linetexel,spacingVariation,25 );
    line1 =[line1  texel];
      
end
blank = ones(size(texel,1),  imageWidth - size(line1,2));
line1 = [line1 blank];

%imshow(line1);

im = line1;

for x = 1: imageHeight
    im = [im;line1];
end



imwrite(im,fileName,'jpg');
imshow(im);
