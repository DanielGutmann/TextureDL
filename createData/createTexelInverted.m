function [texel] = createTexelInverted(height,width,theta,texelLength,type)

%type 1 - line, 2- circle/ellipse 3- filled circle/ellipse
texel = zeros(height,width);


if(type == 3)
    r= texelLength;

    for x = width/2 - r : width/2 + r
        for y = height/2 - r : height/2 +r

        if((x - width/2) ^ 2 +  ( y - height/2) ^2 < r ^ 2 )
            texel(x,round(y)) = 1;
        end

        end 
    end

end

if(type == 1)
    lineLength=texelLength;
    if(theta == pi/2 || theta == 3 * (pi/2) || theta == -pi /2 )
       x = width / 2;
       for y = height/2 - lineLength : height/2 + lineLength
           texel(x,round(y)) = 1;
       end
    else
       for x = width/2 - lineLength : width/2 + lineLength
             y = ( (x - width/2) * tan(theta) ) + (height/2) ;
             texel(x,round(y)) = 1;
       end
        
    end
end


