
%create a sythetic image with two textured region
%radius of dot = r
clear all;
close all;

fileName ='C:\TextureDL\data\Synthesized5.jpg';

%blank texel
linetexel = ones(10,10);
[width height] = size(linetexel);
lineLength=4;
y = height / 2;
orient=pi/4;
for x = width/2 - lineLength : width/2 + lineLength
    linetexel(x,height / 2 + round( tan(orient) * (x-width/2) )) = 0;
end

imshow(linetexel);
%repeat the texel to create a line image
line1 = linetexel;
for y = 1 :60
        line1 = [line1 linetexel];
end

blankim = line1;
for x = 1: 15
    blankim = [blankim;line1];
end



texel = ones(10,10);
[width height] = size(texel);
lineLength=4;
y = height / 2;
orient=0;
for x = width/2 - lineLength : width/2 + lineLength
    texel(x,height / 2 + round( tan(orient) * (x-width/2) )) = 0;
end

%repeat the texel to create an image
blanktexel = ones(10,10);
line1 = linetexel;
for y = 1 :15
  line1 = [line1 linetexel];
end


for y = 16 :45
   line1 = [line1 texel];
end

for y = 46 :60
   line1 = [line1 linetexel];
end



im = line1;
for x = 1: 30
    im = [im;line1];
end



im= [blankim;im ;blankim];


imwrite(im,fileName,'jpg');


imshow(im);
