%create a synthetic image with single textured region
%Put texels at random points

imageWidth = 400;
imageHeight = 400;
texelwidth = 10;
texelheight = 10;
orientation = 3 * pi/4;

im = zeros(imageWidth,imageHeight);
numberOfTexels = 1000;
numberOfpointsAdded = 0;
while(numberOfpointsAdded < numberOfTexels)
    x = randi(imageWidth);
    y = randi(imageHeight);
    
    %check if (x,y) is near boundary
    if(x - texelwidth/2  + 1 > 1 && x + texelwidth/2 < imageWidth && y - texelheight/2 + 1 > 1 && y + texelheight/2 <imageHeight)
        %check for overlapping texels
        window = im(x - texelwidth/2 + 1 :x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 );
        %TODO change this
        if(max(max(window)) == 0)
            texel = createTexel1(texelheight,texelwidth,orientation,0,1);
            im(x - texelwidth/2 + 1:x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 ) = texel;
            numberOfpointsAdded = numberOfpointsAdded + 1;
        end
        
    end
    
    
end





orientation = 0;

im1 = zeros(imageWidth,imageHeight);
numberOfTexels = 1000;
numberOfpointsAdded = 0;
while(numberOfpointsAdded < numberOfTexels)
    x = randi(imageWidth);
    y = randi(imageHeight);
    
    %check if (x,y) is near boundary
    if(x - texelwidth/2  + 1 > 1 && x + texelwidth/2 < imageWidth && y - texelheight/2 + 1 > 1 && y + texelheight/2 <imageHeight)
        %check for overlapping texels
        window = im1(x - texelwidth/2 + 1 :x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 );
        if(max(max(window)) ==0)
            texel = createTexel1(texelheight,texelwidth,orientation,0,1);
            im1(x - texelwidth/2 + 1:x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 ) = texel;
            numberOfpointsAdded = numberOfpointsAdded + 1;
        end
        
    end
    
    
end



%im = [im1;im];


fileName ='C:\TextureDL\Data\RandomSpacing45And0DegreeLine1300TexelInverted.jpg';
imwrite(im,fileName,'jpg');

imshow(im);


