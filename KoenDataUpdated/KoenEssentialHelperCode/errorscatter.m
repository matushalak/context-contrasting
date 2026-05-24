function h = errorscatter(x,y,e,col,baron)

%Scattered errorbar plot, works similar to errorbar, y should be
%a column vector or matrix, col should be a matrix with one row for each column of y.
%Plots in a pre-opened figure
%If y is a matrix then the columns are grouped together around the x
%values, e.g. if y = 3x2 matrix, there will be 3 x values and two groups at
%each x value.
%If baron = 1 then it draws bars instead of scatterd points. Enter col = []
%to use default colors (jetmap)

if size(y,1) ~= length(x)
    error(['y should have as many rows as x values'])
end

ncols = size(y,2);
if nargin < 4 | isempty(col)
    col = jet(ncols);
end

if nargin<5
    baron = 0;
end

xwidth = (mean(diff(x))./ncols)./2;
xbin =  ((xwidth.*ncols)./2).*0.75;

if ncols == 1
    if baron
        g = bar(x,y,'barwidth',xwidth,'FaceColor',col);
        hold on
    end
    h = errorbar(x,y,e,'linestyle','none','Color',col);
    if ~baron
        hold on
        scatter(x,y,[],col,'filled')
    end
   
else
    %Make the x-axis
    xadd = linspace(-xbin,xbin,ncols);
    for n = 1:ncols
        if baron
            g = bar(x+xadd(n),y(:,n),'barwidth',xwidth,'FaceColor',col(n,:));
            hold on
        end
        h(n) = errorbar(x+xadd(n),y(:,n),e(:,n),'linestyle','none','Color',col(n,:));
        hold on
        if ~baron
        scatter(x+xadd(n),y(:,n),[],col(n,:),'filled')
        end
       
    end
    
end

return

