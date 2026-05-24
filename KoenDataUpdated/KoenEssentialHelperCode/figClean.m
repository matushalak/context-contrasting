function figClean(handle)
	
% inputs
if ~exist('handle','var') || isempty(handle)
    handle=gca;
end

% changes
dblFontSize = 12;

title(get(get(handle,'title'),'string'),'FontSize',dblFontSize);
set(handle,'FontSize',dblFontSize,'Linewidth',1.5); % change font size of x/y ticks
set(handle,'TickDir', 'out');
set(handle, 'FontSize', dblFontSize), box off
xlabel(get(get(handle,'xlabel'), 'String'),'FontSize',dblFontSize); %set x-label and change font size
ylabel(get(get(handle,'ylabel'), 'String'),'FontSize',dblFontSize);%set y-label and change font size
fontname(gcf,"Arial")