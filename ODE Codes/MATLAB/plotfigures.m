%% ¶Ô³Æ½â
trial1 = load('.\result-1.mat');
trial2 = load('.\result-11.mat');
t = 0:0.001:100;figure1 = figure(1); set(gcf,'Units','centimeters','Position',[1 5 30 9]); 
axes('Parent',figure1,'Position',[0.08 0.18 0.85 0.73]);
plot(t, trial1.trajectory(:,1)+trial2.trajectory(:,1),'b','linewidth',1.5); hold on
plot([0:2:100],zeros([51,1]),'r--','linewidth',1.5); 
set(gca,'fontname','times new roman', 'fontsize',16);
xlim([-2,102]); legend('DRL solution','Theoretical solution')
grid on; ylim([-1,1])
ylabel('\it{x}_1(t)+\it{x}_2(t)'); xlabel('\itt')
