
imageWidth = 500;
imageHeight = 500;
im = ones(imageWidth,imageHeight);
texelwidth = 8;
texelheight = 8;


xv=[100,300,120,320];
yv=[150,150,400,400];

numberOfTexels = 3300;
numberOfpointsAdded = 0;
while(numberOfpointsAdded < numberOfTexels)
    x = randi(imageWidth);
    y = randi(imageHeight);
    
    %check if (x,y) is near boundary
    if(x - texelwidth/2  + 1 > 1 && x + texelwidth/2 < imageWidth && y - texelheight/2 + 1 > 1 && y + texelheight/2 <imageHeight)
        %check for overlapping texels
        window = im(x - texelwidth/2 + 1 :x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 );
        if(min(min(window)) ==1)
            if inpolygon(x,y,xv,yv) == 1
                texel = createTexel1(texelheight,texelwidth,3*(pi/4),1,0);
                im(x - texelwidth/2 + 1:x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 ) = texel;
                numberOfpointsAdded = numberOfpointsAdded + 1;
            else
                 texel = createTexel1(texelheight,texelwidth,0,1,0);
                im(x - texelwidth/2 + 1:x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 ) = texel;
                numberOfpointsAdded = numberOfpointsAdded + 1;
                
            end
                   
        end
        
    end
    
    
end

fileName ='C:\TextonCode\SingleTexture\Polygon.jpg';
%imwrite(im,fileName,'jpg');


imshow(im);
            