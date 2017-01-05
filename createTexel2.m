function [texel] = createTexel2(height,width,theta,bgrayvalue,fgrayvalue)


texel = bgrayvalue * ones(height,width);


%fwrite(1,sprintf('theta=%0.2f TanTheta=%0.2f \n',theta,tan(theta)));


    lineLength=width;
    if(theta == pi/2 || theta == 3 * (pi/2) || theta == -pi /2 )
       %handle 90 degree and 270 degree separately  becasuse tan 90 is inf 
       x = height / 2;
       for y = 1 : height;
           
           texel(x,y) = fgrayvalue;
           
       end
    else
       for x = 1 : height
             y = ( ( x - width / 2 ) * tan( theta ) ) + ( width/2 );
             %fwrite(1,sprintf('%d %d \n',x,y));
             if( y >= 1 && y <= width && x >= 1 && x <= height )
                texel(x,round(y)) = fgrayvalue;
             end
       end
       
    end
        
end



