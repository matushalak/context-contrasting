%% function to save figure as png, fig, and svg

function func_save_fig(name)

% save as .png for presentations -> try catch since it crashes sometimes
try
    saveas(gcf,[name '.png'])
catch
    saveas(gcf,[name '.png'])
end

% save as .fig and .svg to change in the future
saveas(gcf,[name '.fig'])
saveas(gcf,[name '.svg'])

disp(['Successfully saved figure: ' name])

end