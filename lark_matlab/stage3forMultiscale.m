function [RM2,RM3] = stage3forMultiscale(RM,Q,alpha)

% Significance Test and Non-maxima Suppression for multi-scale
%
% [USAGE]
% [RM2,det_tag,indx,indy,RM3] = stage3forMultiscale(RM,Q,T,alpha)
%
% [RETURNS]
% RM2   : Resemblance map with bounding boxes at detected objects
% RM3   : Resemblance map only including detected objects.
% det_tag : if any object is found, det_tag = 1, otherwise, det_tag = 0;
% indx,indy : spatial locations of found objects
%
% [PARAMETERS]
% RM   : Resemblance map
% Q : query iamge
% alpha     : confidence level

%
% [HISTORY]
% July 08, 2009 : created by Hae Jong

RM1 = RM;
RM2 = zeros(size(RM));
RM3 = zeros(size(RM));

if max(RM(:)) > 0.01
    f_rho = RM(floor(size(Q,1)/2):end-floor(size(Q,1)/2),floor(size(Q,2)/2):end-floor(size(Q,2)/2));
  %  [E_pdf,ind] = hist(f_rho(:),1000);
     [E_pdf, ind] = ksdensity(f_rho(:));
    E_cdf = cumsum(E_pdf/sum(E_pdf));
    detection = find(E_cdf > alpha);
    T_n = ind(detection(1)); %% Parameter for significance testing 2
    if T_n < 0
        T_n = 0;
    end
    x = size(Q,1);
    y = size(Q,2);

    half_x1 = fix(x/2);
    half_y1 = fix(y/2);

    [x1,x2] = meshgrid(-half_y1:half_y1,-half_x1:half_x1);
    tt = x1.^2 + x2.^2;

    kkk = exp(-(0.5/(5)^2) * tt);

    cnt = 0;
    %% Non-maxima suppression

    while max(RM1(:)) >= T_n
%         [max(RM1(:)) T_n]
        cnt = cnt + 1;
        [x_ind,y_ind] = find(RM==max(RM1(:)));

        if numel(x_ind) > 1
            x_ind = x_ind(1);
            y_ind = y_ind(1);
        end
        [x_ind y_ind];
        indx(cnt) = x_ind;
        indy(cnt) = y_ind;
        x_b = x_ind-half_x1;
        y_b = y_ind-half_y1;
        x_e = x_ind+half_x1;
        y_e = y_ind+half_y1;

        if x_b <= 0
            x_b = 1;
            x_e = x_b+size(kkk,1)-1;
        end
        if x_e > floor(size(RM,1))
            x_e = floor(size(RM,1));
            x_b = x_e-size(kkk,1)+1;
        end

        if y_b <= 0
            y_b = 1;
            y_e = y_b+size(kkk,2)-1;
        end
        if y_e > floor(size(RM,2))
            y_e = floor(size(RM,2));
            y_b = y_e-size(kkk,2)+1;
        end
%         RM3(x_b:x_e,y_b:y_e) = RM2(x_b:x_e,y_b:y_e);
        RM2(x_b:x_e,y_b:y_e) = RM1(x_b:x_e,y_b:y_e).*kkk(1:size(RM1(x_b:x_e,y_b:y_e),1),1:size(RM1(x_b:x_e,y_b:y_e),2));
        RM3(x_b:x_e,y_b:y_e) = RM1(x_b:x_e,y_b:y_e);

        %         RM2 = DrawBox_bold(RM2,x_b,y_b,x_e,y_e,max(RM1(:)));
        RM2 = DrawBox(RM2,x_b,y_b,x_e,y_e,max(RM1(:)));

        RM1(x_b:x_e,y_b:y_e) = 0;

    end
    det_tag = 1;
else

    det_tag = 0;
    indx = 0;
    indy = 0;
end


function img = DrawBox(img, x0,y0,x1,y1,value)

img = func_DrawLine(img, x0, y0, x0, y1, value);
img = func_DrawLine(img, x0, y1, x1, y1, value);
img = func_DrawLine(img, x1, y1 ,x1, y0, value);
img = func_DrawLine(img, x1, y0, x0, y0, value);

function Img = func_DrawLine(Img, X0, Y0, X1, Y1, nG)
% Connect two pixels in an image with the desired graylevel
%
% Command line
% ------------
% result = func_DrawLine(Img, X1, Y1, X2, Y2)
% input:    Img : the original image.
%           (X1, Y1), (X2, Y2) : points to connect.
%           nG : the gray level of the line.
% output:   result
%
% Note
% ----
%   Img can be anything
%   (X1, Y1), (X2, Y2) should be NOT be OUT of the Img
%
%   The computation cost of this program is around half as Cubas's [1]
%   [1] As for Cubas's code, please refer  
%   http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=4177  
%
% Example
% -------
% result = func_DrawLine(zeros(5, 10), 2, 1, 5, 10, 1)
% result =
%      0     0     0     0     0     0     0     0     0     0
%      1     1     1     0     0     0     0     0     0     0
%      0     0     0     1     1     1     0     0     0     0
%      0     0     0     0     0     0     1     1     1     0
%      0     0     0     0     0     0     0     0     0     1
%
%
% Jing Tian Oct. 31 2000
% scuteejtian@hotmail.com
% This program is written in Oct.2000 during my postgraduate in 
% GuangZhou, P. R. China.
% Version 1.0

Img(X0, Y0) = nG;
Img(X1, Y1) = nG;
if abs(X1 - X0) <= abs(Y1 - Y0)
   if Y1 < Y0
      k = X1; X1 = X0; X0 = k;
      k = Y1; Y1 = Y0; Y0 = k;
   end
   if (X1 >= X0) & (Y1 >= Y0)
      dy = Y1-Y0; dx = X1-X0;
      p = 2*dx; n = 2*dy - 2*dx; tn = dy;
      while (Y0 < Y1)
         if tn >= 0
            tn = tn - p;
         else
            tn = tn + n; X0 = X0 + 1;
         end
         Y0 = Y0 + 1; Img(X0, Y0) = nG;
      end
   else
      dy = Y1 - Y0; dx = X1 - X0;
      p = -2*dx; n = 2*dy + 2*dx; tn = dy;
      while (Y0 <= Y1)
         if tn >= 0
            tn = tn - p;
         else
            tn = tn + n; X0 = X0 - 1;
         end
         Y0 = Y0 + 1; Img(X0, Y0) = nG;
      end
   end
else if X1 < X0
      k = X1; X1 = X0; X0 = k;
      k = Y1; Y1 = Y0; Y0 = k;
   end
   if (X1 >= X0) & (Y1 >= Y0)
      dy = Y1 - Y0; dx = X1 - X0;
      p = 2*dy; n = 2*dx-2*dy; tn = dx;
      while (X0 < X1)
         if tn >= 0
            tn = tn - p;
         else
            tn = tn + n; Y0 = Y0 + 1;
         end
         X0 = X0 + 1; Img(X0, Y0) = nG;
      end
   else
      dy = Y1 - Y0; dx = X1 - X0;
      p = -2*dy; n = 2*dy + 2*dx; tn = dx;
      while (X0 < X1)
         if tn >= 0
            tn = tn - p;
         else
            tn = tn + n; Y0 = Y0 - 1;
         end
         X0 = X0 + 1; Img(X0, Y0) = nG;
      end
   end
end
