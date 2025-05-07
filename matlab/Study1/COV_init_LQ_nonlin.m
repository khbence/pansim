clear
close all
clc
%% Log parameters
id = sprintf('result_%s', datetime('now', 'Format','dd-MM-yy_HH:mm'));
fprintf('Script started, result id: %s\n', id);

%% Simulation parameters

ref_sample_time = -1; % REDUNDANT??
ratetrans_sample_time = 0.02;
input_sample_time = 0.1;

tf = 365; % days

%% Reference signal

dt = 0.1; % reference sampling time, REDUNDANT??
t_sim = 0:dt:tf;

REF_FILE = load("res2.mat");
x_ref = [1:length(REF_FILE.Ihat_daily); REF_FILE.Ihat_daily]';

spline_Iref = spline(x_ref(:, 1), x_ref(:, 2)); %create the funcion of the reference curve
spline_Irefd = fnder(spline_Iref, 1);           %calculate the first derivate
spline_Irefdd = fnder(spline_Iref, 2);          %calculate the second derivate

Iref = [t_sim.', ppval(spline_Iref, t_sim).'];
Irefd = [t_sim.', ppval(spline_Irefd, t_sim).'];
Irefdd = [t_sim.', ppval(spline_Irefdd, t_sim).'];

%% Simulation model (SLPIAHRD)
alpha = 1/2.5;
p = 1/3;
beta = 1/3;
delta = 0.75;
q = 0.6;
p_I = 1/4;
p_A = 1/4;
eta = 0.076;
h = 1/10;
mu = 0.145;
R_0 = beta * (1 / p + q / p_I + delta * (1 - q) / p_A);

% Initial conditions
N=1; % Population normalized
L0 = 5.102 * 10^ - 6;
P0 = 5.102/3 * 10^ - 6;
I0 =  5.102/3 * 10^ - 6;
A0 = 5.102/3 * 10^ - 6;
H0 = 0;
R0 = 0;
D0 = 0;
S0 = N - (L0 + P0 + A0 + I0 + R0 + H0 + D0);
x0 = [S0; L0; P0; A0; I0; R0; H0; D0];

%% Control design model & feedback
k1 = 0.5;               % nominal value, time-varying input when controlled
k2 = 0.37;              % nominal value, valid 0.3-0.4
k3 = 0.1429;            % nominal value, valid ~0.15
k2_error = 1;           % error multiplier 
k3_error = 1;           % error multiplier
u_min = 0;              % input saturation (lower)
u_max = k1;             % input saturation (upper)

A = [0 1 0; 0 0 1; 0 0 0];
B = [0; 0; 1];
p_fp = [-0.1; -0.12; -0.14]*3.39322177189533; % *2
%p_fp = [-0.1; -0.12; -0.14]*1.25;
% p_fp = [-0.15 -0.1 -0.05] * 1.5;
feedback_pars = -place(A, B, p_fp);
%feedback_pars = -lqr(A, B, 0.1*eye(3), 0.01);

c1 = feedback_pars(1);
c2 = feedback_pars(2);
c3 = feedback_pars(3);

fprintf('Calculated feedback gains (PID): %f %f %f\n', c1, c2, c3);

%ddr+(-v+(k2*k3+k2^2)*E-k3^2*I)/(k2*S*I)

z1_start = abs(x_ref(1, 2) - I0);

%% Extended Kalman Filter parameters


ekf_sample_time = 0.02; % TODO: modify transfer functions!!
switchtime = 30;

% EKF SEIR:
ekf_seir_initialstate = [0 5*1e-6 1e-6 0];
ekf_seir_initialstate(1) = 1 - sum(ekf_seir_initialstate);
ekf_seir_processcov = diag([0 0 0 0]);
ekf_seir_initcov = 1;
ekf_seir_measurecov = eps;


% EKF SLPIAHRD
ekf_slpiahrd_initialstate = [0 1e-6 1e-7 1e-7 1e-7 0 0 0];
ekf_slpiahrd_initialstate(1) = 1 - sum(ekf_slpiahrd_initialstate);
ekf_slpiahrd_processcov = diag([0 0 0 0 0 0 0 0]);%diag([1e-9 1e-9 0 0 0 1e-6 0 0]);
ekf_slpiahrd_initcov = 1;
ekf_slpiahrd_measurecov = eps;
ekf_slpiahrd_initialstate = x0;


%% Run the simulation

% Multiple simulations
tic
counter = 1;
% k2_error_range = [0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2];
% k3_error_range = [0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2];
% 
% k2_error_range = [0.5 1 2];
% k3_error_range = [0.5 1 2];


% k2_error_range = [0.4 1 2.5];
% k3_error_range = [0.4 1 2.5];

% k2_error_range = [0.5 0.7 0.9 1 1.2 1.5 2];
% k3_error_range = [0.5 0.7 0.9 1 1.2 1.5 2];
% k2_error_range = [0.2 0.5 0.8 1 1.2 1.5];
% k3_error_range = [0.2 0.5 0.8 1 1.2 1.5];

% allres = cell(length(k2_error_range), length(k3_error_range));
% 
% for k2_i = 1:length(k2_error_range)
%     for k3_i = 1:length(k3_error_range)
%         k2_error = k2_error_range(k2_i);
%         k3_error = k3_error_range(k3_i);
%         fprintf('#%d: errors (k2, k3) = (%f %f)\n', counter, k2_error, k3_error);
% 
%         load_system('COV_sim_LQ_nonlin');
%         result = sim('COV_sim_LQ_nonlin');
%         allres{k2_i, k3_i} = result;
%         counter = counter + 1;
%         toc
%     end
% end

% One simulation only
load_system('COV_sim_LQ_nonlin');
result = sim('COV_sim_LQ_nonlin');

save(sprintf('result/%s.mat', id));
fprintf('Result saved!\n');



%% Figure generation

plttime = 1:tf;

figure(99); defcolors = get(gca,'colororder'); % default colors of a plot
NPop = 9.8e6; % Population
XDates = [datetime(2020,08,15):datetime(2021,08,14)];


close all;
set(0,'DefaultFigureWindowStyle','docked') %% -- for experiments
% set(0,'DefaultFigureWindowStyle','normal') %% for figure generation

% result = allres{3,2}; % nominal result to show on plots
% pltres = allres(1:end, 1:end);

% FIG1: Reference tracking
f = figure(1); f.RendererMode = 'auto'; f.Renderer = 'painters';
clf; hold on; grid on;
title('Reference tracking result')
plot(plttime, NPop*interp1(result.I_v_ref.time, result.I_v_ref.signals.values(:, 2), plttime), 'r', 'Linewidth', 2) % reference
plot(plttime, NPop*interp1(result.I_v_ref.time, result.I_v_ref.signals.values(:, 1), plttime), 'Color', defcolors(1, :), 'Linewidth', 2) % nominal
legend('I_{REF}', 'I (SEIR)');

xlim([0 tf])
xlabel('Time [days]')

ymax = 2.1e5;
yyaxis left
ylim([0 ymax]);
yticks(0:ymax/5:ymax);
ylabel('Nr. of infected')
yyaxis right
yymax = 2.5;
ylim([0 ymax/NPop*100]);
yticks(0:yymax/5:yymax);
ylabel('% of population');
ax = gca;
ax.YAxis(1).Color = 'k';
ax.YAxis(2).Color = 'b';



% FIG2 : Input
f = figure(2); f.RendererMode = 'auto'; f.Renderer = 'painters';
clf; hold on; grid on;
title('Input (of the simulation model)');
plot(result.system_input.Time, result.system_input.Data, 'LineWidth',2);
%plot(result.raw_system_input.Time, result.raw_system_input.Data, 'LineWidth',2);
ylim_low = u_min-(u_max-u_min)*0.2;
ylim_high = u_max+(u_max-u_min)*0.2;
ylim([ylim_low, ylim_high]);
legend('input (\beta)');


% FIG3: EKF SEIR
f = figure(3); f.RendererMode = 'auto'; f.Renderer = 'painters';
clf; hold on; grid on;
title('EKF SEIR and true SEIR states')
plot(result.seir_states.time, result.seir_states.signals.values, '--');
ax = gca; ax.ColorOrderIndex = 1;
plot(result.ekf_seir.Time, result.ekf_seir.Data, 'Linewidth', 2)
xlim([0 tf])
legend('S', 'E', 'I', 'R')
xlabel('Time [days]')
ylabel('prop. of population')

% FIG4: EKF SLPIAHRD
f = figure(4); f.RendererMode = 'auto'; f.Renderer = 'painters';
clf; hold on; grid on;
title('EKF SLPIAHRD and true SLPIAHRD')
plot(result.slpiahrd_states.Time, result.slpiahrd_states.Data, '--');
ax = gca; ax.ColorOrderIndex = 1;
plot(result.ekf_slpiahrd.Time, result.ekf_slpiahrd.Data, 'Linewidth', 2)
xlim([0 tf])
legend('S', 'L','P', 'I', 'A', 'H', 'R', 'D', 'EKF S', 'EKF L');
xlabel('Time [days]')
ylabel('prop. of population')



% pltres_ref = zeros(numel(pltres), numel(plttime));
% for res_i = 1:numel(pltres)
%     res = pltres{res_i};
%     pltres_ref(res_i, :) = interp1(res.I_v_ref.time, res.I_v_ref.signals.values(:, 1), plttime);
% end
% 
% pltres_inputs = zeros(numel(pltres), numel(plttime));
% for res_i = 1:numel(pltres)
%     res = pltres{res_i};
%     pltres_inputs(res_i, :) = interp1(res.system_input.Time, movmean(res.system_input.Data,3), plttime);
% end
% 
% pltres_ekf_states = zeros(0,numel(plttime),4);
% for res_i = 1:numel(pltres)
% res = pltres{res_i};
% pltres_ekf_states(res_i, :, 1) = interp1(res.ekf_seir.time, res.ekf_seir.Data(:, 1), plttime);
% pltres_ekf_states(res_i, :, 2) = interp1(res.ekf_seir.time, res.ekf_seir.Data(:, 2), plttime);
% pltres_ekf_states(res_i, :, 3) = interp1(res.ekf_seir.time, res.ekf_seir.Data(:, 3), plttime);
% pltres_ekf_states(res_i, :, 4) = interp1(res.ekf_seir.time, res.ekf_seir.Data(:, 4), plttime);
% end
% 
% % FIG5: reference tracking with param uncertainty
% f = figure(5); f.RendererMode = 'auto'; f.Renderer = 'painters';
% set(f, 'Position', 1.2*[1 1 700 150]);
% clf; hold on; grid on;
% %title('Reference tracking result')
% % plot(plttime, NPop*interp1(result.I_v_ref.time, result.I_v_ref.signals.values(:, 2), plttime), 'r', 'Linewidth', 2) % reference
% % plot(plttime, NPop*interp1(result.I_v_ref.time, result.I_v_ref.signals.values(:, 1), plttime), 'Color', defcolors(1, :), 'Linewidth', 2) % nominal
% % plotmeanandstd(plttime, NPop*pltres_ref, defcolors(1, :));
% plot(datenum(XDates), NPop*interp1(result.I_v_ref.time, result.I_v_ref.signals.values(:, 2), plttime), 'r', 'Linewidth', 3) % reference
% plot(datenum(XDates), NPop*interp1(result.I_v_ref.time, result.I_v_ref.signals.values(:, 1), plttime), 'Color', defcolors(1, :), 'Linewidth', 2) % nominal
% plotmeanandstd(datenum(XDates), NPop*pltres_ref, defcolors(1, :));
% %xlim([0 tf])
% legend('I_{REF}', 'I (SEIR)')
% % xlabel('Time [days]')
% 
% xlim([datenum(XDates(1)) datenum(XDates(end))]);
% xticks(datenum(XDates(1:30:end)));
% datetick('x','mm/yy','keepticks');
% xlabel('Time (month/year)');
% 
% ymax = 1.2e5;
% yyaxis left
% ylim([0 ymax]);
% yticks(0:ymax/12:ymax);
% ylabel('Nr. of infected')
% yyaxis right
% yymax = ymax/1e5;
% ylim([0 ymax/NPop*100]);
% yticks(0:yymax/6:yymax);
% ylabel('% of population');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% ax.YAxis(2).Color = 'b';
% 
% % FIG6: input with param uncertainty
% f = figure(6); f.RendererMode = 'auto'; f.Renderer = 'painters';
% set(f, 'Position', 1.2*[1 1 700 150]);
% clf; hold on; grid on;
% %title('Input (of the simulation model)');
% plot(datenum(XDates), interp1(result.system_input.Time, result.system_input.Data, plttime), 'LineWidth',2); % nominal
% plotmeanandstd(datenum(XDates), pltres_inputs, [0, 0.4470, 0.7410]);
% 
% yyaxis left
% ylim([u_min-(u_max-u_min)*0.1 u_max+(u_max-u_min)*0.1])
% yticks(0:0.05:0.5);
% ylabel('Simulation model input')
% yyaxis right
% ylabel('|')
% yticks([])
% 
% legend('input (\beta)');
% ax = gca;
% ax.YAxis(1).Color = 'k';
% ax.YAxis(2).Color = [1 1 1];
% 
% xlim([datenum(XDates(1)) datenum(XDates(end))]);
% xticks(datenum(XDates(1:30:end)));
% datetick('x','mm/yy','keepticks');
% xlabel('Time (month/year)');
% 
% % % FIG7: ekf seir states with param uncertainty
% % f = figure(7); f.RendererMode = 'auto'; f.Renderer = 'painters';
% % clf; hold on; grid on;
% % title('EKF SEIR and true SEIR states')
% % plot(result.seir_states.time, result.seir_states.signals.values, '--'); % reference
% % ax = gca; ax.ColorOrderIndex = 1;
% % plot(result.ekf_seir.Time, result.ekf_seir.Data, 'Linewidth', 2)
% % defcolors = get(gca,'colororder');
% % plotmeanandstd(plttime, pltres_ekf_states(:, :, 1), defcolors(1, :));
% % plotmeanandstd(plttime, pltres_ekf_states(:, :, 2), defcolors(2, :));
% % plotmeanandstd(plttime, pltres_ekf_states(:, :, 3), defcolors(3, :));
% % plotmeanandstd(plttime, pltres_ekf_states(:, :, 4), defcolors(4, :));
% % xlim([0 tf])
% % legend('S', 'E', 'I', 'R')
% % xlabel('Time [days]')
% % ylabel('prop. of population')
% 
% 
% % FIG8: cost of interference
% C = zeros(size(allres));
% for k2i = 1:numel(k2_error_range)
%     for k3i = 1:numel(k3_error_range)
%         res = allres{k2i, k3i};
%         C(k2i, k3i) = trapz(res.system_input.Time, (u_max - res.system_input.Data).^2);
%     end
% end
% [XX, YY] = meshgrid(k2_error_range, k3_error_range);
% 
% 
% f = figure(8); f.RendererMode = 'auto'; f.Renderer = 'painters';
% set(f, 'Position', [1 1 450 300]);
% clf; hold on; grid on;
% surf(XX, YY, C);
% colorbar
% caxis([min(C(:)),min(C(:)) + 0.1*(max(C(:)) - min(C(:)))]);
% % set(gca, 'clim', [min(C(:)), min(C(:)) + 0.25*(max(C(:)) - min(C(:)))]);
% % zticks(min(C(:)):0.05:max(C(:))+0.1)
% xticks(k2_error_range)
% yticks(k3_error_range)
% view(-45,45)
% xlabel('k_2 multiplier')
% ylabel('k_3 error')
% zlabel('cost')
% ztickformat('%,.2f')
%plot3(range(5), range(5), C(5,5), 'r*');
%%
% for fid=[5 6 8]
%     imagename = sprintf('figs/%s_fig%d.pdf', id, fid);
%     save_image(imagename, fid);
% end


% controll_q = trapz(result.c_q_tmp.time, result.c_q_tmp.data)
% cost_of_interference = trapz(result.coi_tmp.time, result.coi_tmp.data)
%% Helper functions

function save_image(imagename, k)
    figure(k)
    print(imagename, '-dpdf');
    system(sprintf('pdfcrop %s %s', imagename, imagename));
end

function plotmeanandstd(x, ys, varargin)
m = mean(ys);
s = std(ys);
r2 = fillbetween(x, m-2*s, m+2*s, varargin{:}, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
r1 = fillbetween(x, m-1*s, m+1*s, varargin{:}, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end

function [r] = fillbetween(x, y_low, y_high, varargin)
x2 = [x, fliplr(x)];
inBetween = [y_low, fliplr(y_high)];
r = fill(x2, inBetween, varargin{:});
end

function [dates] = simtime2date(simtime)

end


%%% ÖTLET: mi a búbánatnak becslem a sima nemlin rendszert, ha becsülhetném
%%% helyette a linearizált rendszer állapotát amire utána a kontrollert
%%% tervezem??? és még kalman filter sem kéne, csak egy kutyaközönséges
%%% state observer. Megfigyelhető? mert ha igen akkor hawaii