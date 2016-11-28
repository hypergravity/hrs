cd ~/PycharmProjects/hrs/hrs/figures/

%%
w = load('../calibration/w.dat');
% w = flipud(w);

%%
[mx, my] = meshgrid(1:4096, 1:103);

figure(); grid on;
surface(mx, my, w, 'EdgeColor', 'none');
view(-136,22);

% colormap(jet)
xlabel('CCD X-coordinate')
ylabel('Extracted order number')
zlabel('Wavelength (A)')

set(gca, 'clim', [3742, 10622], ...
    'xlim', [-.5 4096.5], ...
    'ylim', [-.5 103.5],...
    'zlim', [3500, 11000])

colorbar

%%
set(gcf, 'papersize', [8 6], 'paperposition', [0 0 8 6])
print(gcf, '-dpdf', './visualize_wavelength_calibration.pdf')
print(gcf, '-dpsc', './visualize_wavelength_calibration.eps')

%%
colormap(jet)
set(gcf, 'papersize', [8 6], 'paperposition', [0 0 8 6])
print(gcf, '-dpdf', './visualize_wavelength_calibration_jet.pdf')
print(gcf, '-dpsc', './visualize_wavelength_calibration_jet.eps')

%%
delete(gcf)