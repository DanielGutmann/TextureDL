function [texel] = createTexel(height,width,theta,texelLength,type,grayvalue)

%type 1 - line, 2- circle/ellipse 3- filled circle/ellipse
texel = grayvalue *ones(height,width);
fwrite(1,sprintf('Height %d Width %d',height, width));

heightByTwo = floor(height / 2);
widthByTwo = floor(width / 2);
if(type == 3)
    r= grayvalue;

    for x = widthByTwo - r : widthByTwo + r
        for y = heightByTwo - r : heightByTwo +r

        if((x - widthByTwo) ^ 2 +  ( y - heightByTwo) ^2 < r ^ 2 )
            texel(x,round(y)) = 0;
        end

        end 
    end

end

fwrite(1,sprintf('theta=%0.2f TanTheta=%0.2f \n',theta,tan(theta)));
if(type == 1)
    lineLength=texelLength;
    if(theta == pi/2 || theta == 3 * (pi/2) || theta == -pi /2 )
       x = widthByTwo;
       for y = heightByTwo - lineLength : heightByTwo + lineLength
           
           texel(x,round(y)) = 0;
           
%            texel(x-1,round(y)) = 0;
%            
%            texel(x+1,round(y)) = 0;
       end
    else
       for x = widthByTwo - lineLength : widthByTwo + lineLength
             y = ( (x - widthByTwo) * tan(theta) ) + (heightByTwo) ;
             fwrite(1,sprintf('%d %d \n',x,y));
             if(y >=1 && y <= width && x >=1 && x <= width)
                texel(x,round(y)) = 0;
             end
%              if( round(y) -1  > 0)
%              texel(x,round(y) -1) = 0;
%              end
%              
%               if( round(y) -2  > 0)
%              texel(x,round(y) -2) = 0;
%              end
%              if( round(y) + 1 < width )
%              texel(x,round(y) +1) = 0;
%              end
%              
%              if( round(y) + 2 < width )
%              texel(x,round(y) +2) = 0;
%              end
       end
        
    end
end


