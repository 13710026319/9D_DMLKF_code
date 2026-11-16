classdef CPF < handle
    % CPF - Centralized Particle Filter (集中式粒子滤波)
    % 9I 维联合状态 (位置3I、速度3I、姿态SO(3)^I) 上的蒙特卡洛滤波。
    % 预测步严格对应文档 Section II；更新步在文档 Section III 基础上采用
    % "自适应似然退火 + MCMC 移动"，以应对初始误差大导致的粒子贫化。

    properties
        Vehicle_num
        Anchor_num
        anchors
        dt_imu
        g_vec

        % 传感器噪声参数 (由外部脚本直接赋真值)
        IMU_Sigma_a
        IMU_Sigma_w
        UWB_sigma_anc
        UWB_sigma_rel

        % 粒子滤波核心参数
        Np
        neff_ratio       = 0.5   % 跨历元重采样触发阈值比例 (制御重采样频率)
        temper_ess_ratio = 0.5   % 退火内部单步 Δγ 的 ESS 阈值比例 (控制退火步长)
        rough_coeff      = 0.10  % (仅旧批处理路径) roughening 系数
        min_pos_jitter   = 0.001 % 位置最小协方差保底噪底 (m)
        min_vel_jitter   = 0.002 % 速度最小协方差保底噪底 (m/s)
        min_att_jitter   = 0.0005% 姿态最小协方差保底噪底 (rad)

        prior_pos_std = [0; 0; 0]
        prior_vel_std = 0
        prior_att_std = 0

        % ---- 似然退火 + MCMC 内部参数 (max_stages 已固化，无需外部传入) ----
        use_tempering   = true   % true: 自适应退火 SMC；false: 旧批处理路径
        n_mcmc          = 3      % 每个退火阶段的 MH 迭代次数
        mcmc_scale      = 0.5    % MH 随机游走步长缩放 (自适应调整)
        mcmc_acc_target = 0.25   % 目标接受率
        max_stages      = 300    % 内部固化：单历元最大退火阶段数上限
        min_dgamma      = 1e-6   % 单阶段最小 γ 增量
        last_num_stages = 0      % 诊断：上一历元退火阶段数
        last_accept     = NaN    % 诊断：上一历元平均 MH 接受率

        % ---- 贫化监控与统计 ----
        batch_size             = 10
        depletion_streak_limit = 3
        depletion_streak       = 0
        update_count           = 0
        resample_count         = 0
        severe_count           = 0

        % 粒子内部状态
        P_p
        P_v
        P_R
        P_logw

        % MMSE 后验估计
        p
        v
        R
    end

    methods
        function obj = CPF(Vehicle_num, Anchor_num, anchors, dt_imu, p0, v0, R0, Noise)
            obj.Vehicle_num = Vehicle_num;
            obj.Anchor_num  = Anchor_num;
            obj.anchors     = anchors;
            obj.dt_imu      = dt_imu;
            obj.g_vec       = [0; 0; -9.81];

            obj.IMU_Sigma_a   = (0.05)^2 * eye(3);
            obj.IMU_Sigma_w   = (0.005)^2 * eye(3);
            obj.UWB_sigma_anc = 0.01;
            obj.UWB_sigma_rel = 0.01;

            if nargin >= 8 && ~isempty(Noise) && isstruct(Noise)
                if isfield(Noise, 'IMU_Sigma_a'),   obj.IMU_Sigma_a   = Noise.IMU_Sigma_a;   end
                if isfield(Noise, 'IMU_Sigma_w'),   obj.IMU_Sigma_w   = Noise.IMU_Sigma_w;   end
                if isfield(Noise, 'UWB_sigma_anc'), obj.UWB_sigma_anc = Noise.UWB_sigma_anc; end
                if isfield(Noise, 'UWB_sigma_rel'), obj.UWB_sigma_rel = Noise.UWB_sigma_rel; end
            end

            obj.Np = 1000;
            obj.p  = p0(:);
            obj.v  = v0(:);
            obj.R  = R0;

            obj.set_particle_count(obj.Np);
        end

        function set_particle_count(obj, Np_new)
            obj.Np = Np_new;
            obj.init_prior(obj.prior_pos_std, obj.prior_vel_std, obj.prior_att_std);
        end

        function init_prior(obj, pos_std_vec, vel_std, att_std_rad)
            obj.prior_pos_std = pos_std_vec(:);
            obj.prior_vel_std = vel_std;
            obj.prior_att_std = att_std_rad;

            I_num   = obj.Vehicle_num;
            Np_curr = obj.Np;

            obj.P_p    = repmat(obj.p(:), 1, Np_curr);
            obj.P_v    = repmat(obj.v(:), 1, Np_curr);
            obj.P_R    = repmat(obj.R, 1, 1, 1, Np_curr);
            obj.P_logw = log(1 / Np_curr) * ones(1, Np_curr);

            if any(pos_std_vec > 0) || vel_std > 0 || att_std_rad > 0
                pos_std_full = repmat(pos_std_vec(:), I_num, 1);

                noise_p = pos_std_full .* randn(3 * I_num, Np_curr);
                noise_p = noise_p - mean(noise_p, 2);
                obj.P_p = obj.P_p + noise_p;

                noise_v = vel_std * randn(3 * I_num, Np_curr);
                noise_v = noise_v - mean(noise_v, 2);
                obj.P_v = obj.P_v + noise_v;

                if att_std_rad > 0
                    for i = 1:I_num
                        noise_phi = att_std_rad * randn(3, Np_curr);
                        noise_phi = noise_phi - mean(noise_phi, 2);
                        Ri  = pagemtimes(repmat(obj.R(:, :, i), 1, 1, Np_curr), ...
                                         obj.exp_SO3_batch(noise_phi));
                        obj.P_R(:, :, i, :) = reshape(Ri, 3, 3, 1, Np_curr);
                    end
                end
            end
            obj.update_mmse_estimate();
        end

        % =========================================================
        % 预测步：文档 Section II (式 6-12)
        % =========================================================
        function predict(obj, acc_m, gyro_m)
            if size(obj.P_p, 2) ~= obj.Np
                obj.set_particle_count(obj.Np);
            end

            I_num = obj.Vehicle_num;
            tau   = obj.dt_imu;
            N     = obj.Np;

            L_a = chol((obj.IMU_Sigma_a + obj.IMU_Sigma_a') / 2, 'lower');
            L_w = chol((obj.IMU_Sigma_w + obj.IMU_Sigma_w') / 2, 'lower');

            for i = 1:I_num
                idx3 = 3*i-2 : 3*i;
                ai = acc_m(:, i);
                wi = gyro_m(:, i);

                e_a = L_a * randn(3, N);
                e_w = L_w * randn(3, N);

                Ri    = reshape(obj.P_R(:, :, i, :), 3, 3, N);
                v_old = obj.P_v(idx3, :);
                p_old = obj.P_p(idx3, :);

                phi = tau * (wi - e_w);
                a_b = ai - e_a;
                a_n = reshape(pagemtimes(Ri, reshape(a_b, 3, 1, N)), 3, N) + obj.g_vec;

                Rn = pagemtimes(Ri, obj.exp_SO3_batch(phi));
                obj.P_R(:, :, i, :) = reshape(Rn, 3, 3, 1, N);
                obj.P_v(idx3, :) = v_old + tau * a_n;
                obj.P_p(idx3, :) = p_old + tau * v_old + 0.5 * tau^2 * a_n;
            end

            obj.update_mmse_estimate();
        end

        % =========================================================
        % 更新步：文档 Section III
        % =========================================================
        function update(obj, uwb_anc, uwb_rel)
            if size(obj.P_p, 2) ~= obj.Np
                obj.set_particle_count(obj.Np);
            end
            obj.update_count = obj.update_count + 1;

            meas = obj.collect_measurements(uwb_anc, uwb_rel);
            if isempty(meas.val), return; end

            if obj.use_tempering
                epoch_severe = obj.update_tempered(meas);
            else
                epoch_severe = obj.update_batched(meas);
            end

            if epoch_severe
                obj.severe_count     = obj.severe_count + 1;
                obj.depletion_streak = obj.depletion_streak + 1;
                if obj.depletion_streak > obj.depletion_streak_limit
                    error('CPF:SevereParticleDepletion', ...
                          ['连续 %d 次历元出现粒子严重贫化警告 (UWB 更新序号 %d, 累计 %d 次)，' ...
                           '已终止本次运行以便调整参数。'], ...
                          obj.depletion_streak, obj.update_count, obj.severe_count);
                end
            else
                obj.depletion_streak = 0;
            end
        end

        function meas = collect_measurements(obj, uwb_anc, uwb_rel)
            I_num = obj.Vehicle_num;
            [ia, la] = find(~isnan(uwb_anc));
            va = uwb_anc(sub2ind(size(uwb_anc), ia, la));

            mask = ~isnan(uwb_rel);
            mask(1:I_num+1:end) = false;
            [ir, jr] = find(mask);
            vr = uwb_rel(sub2ind(size(uwb_rel), ir, jr));

            meas.val  = [va; vr];
            meas.type = [ones(numel(va), 1); 2 * ones(numel(vr), 1)];
            meas.i    = [ia; ir];
            meas.j    = [la; jr];
            meas.var  = [obj.UWB_sigma_anc^2 * ones(numel(va), 1); ...
                         obj.UWB_sigma_rel^2 * ones(numel(vr), 1)];
        end

        function ll = loglik(obj, Pp, meas)
            M  = numel(meas.val);
            ll = zeros(1, size(Pp, 2));
            for m = 1:M
                ii  = meas.i(m);
                pi_ = Pp(3*ii-2:3*ii, :);
                if meas.type(m) == 1
                    diff = pi_ - obj.anchors(meas.j(m), :)';
                else
                    jj   = meas.j(m);
                    diff = pi_ - Pp(3*jj-2:3*jj, :);
                end
                d  = sqrt(sum(diff.^2, 1));
                ll = ll - 0.5 * (meas.val(m) - d).^2 / meas.var(m);
            end
        end

        % ---------------- 自适应似然退火 SMC (已修复 γ=1 退出与双阈值分离) ----------------
        function severe = update_tempered(obj, meas)
            I_num = obj.Vehicle_num;
            N     = obj.Np;
            nP    = 3 * I_num;

            % 分离两个阈值：ess_temp_tgt 控制退火步长，ess_res_tgt 控制重采样触发
            ess_temp_tgt = obj.temper_ess_ratio * N;
            ess_res_tgt  = obj.neff_ratio * N;
            severe       = false;

            logw = obj.P_logw(:)';
            logw = logw - max(logw);
            logw = logw - log(sum(exp(logw)));

            logL = obj.loglik(obj.P_p, meas);

            % [快速通道]：若一步直接加满 γ=1 后，ESS 仍高于重采样阈值 ess_res_tgt，
            % 则无需退火、无需切空间高斯拟合与 MCMC 扰动，直接按纯正 SIR 粒子滤波更新权重！
            logw_full = logw + logL;
            logw_full = logw_full - max(logw_full);
            logw_full = logw_full - log(sum(exp(logw_full)));
            if obj.ess_of(logw_full) >= ess_res_tgt
                obj.P_logw = log(max(exp(logw_full), realmin));
                obj.update_mmse_estimate();
                obj.last_num_stages = 1;
                obj.last_accept     = NaN;
                return;
            end

            % [退火通道]：当直接更新会导致 ESS < ess_res_tgt 时，启动切空间退火 + MCMC
            obj.update_mmse_estimate();
            [X, Rref] = obj.to_chart();

            w0     = exp(logw);
            mu0    = X * w0(:);
            Xc     = X - mu0;
            fvar   = obj.floor_variance();
            Sigma0 = (Xc .* w0) * Xc';
            Sigma0 = (Sigma0 + Sigma0') / 2 + diag(fvar);
            L0     = chol(Sigma0, 'lower');
            optsLT = struct('LT', true);
            logprior = @(Xm) -0.5 * sum(linsolve(L0, Xm - mu0, optsLT).^2, 1);

            lp      = logprior(X);
            acc_sum = 0;
            acc_cnt = 0;

            % 若起始先验权重已低于重采样阈值，先重采样归整
            if obj.ess_of(logw) < ess_res_tgt
                idx  = obj.systematic_resample(exp(logw));
                X    = X(:, idx); logL = logL(idx); lp = lp(idx);
                logw = -log(N) * ones(1, N);
                obj.resample_count = obj.resample_count + 1;
                [X, logL, lp, acc] = obj.mcmc_move(X, logL, lp, 0, meas, logprior, fvar);
                acc_sum = acc_sum + acc; acc_cnt = acc_cnt + 1;
            end

            gamma   = 0;
            n_stage = 0;
            while gamma < 1
                n_stage = n_stage + 1;
                r = 1 - gamma;

                if n_stage >= obj.max_stages
                    dg = r;
                    severe = true;
                else
                    dg = obj.next_gamma(logw, logL, r, ess_temp_tgt);
                    dg = min(r, max(dg, obj.min_dgamma));
                end

                logw = logw + dg * logL;
                logw = logw - max(logw);
                logw = logw - log(sum(exp(logw)));
                gamma = gamma + dg;
                if r - dg < 1e-12, gamma = 1; end

                if gamma < 1
                    % 中间退火阶段：重采样 + 在当前温度 gamma 下执行 MCMC 移动
                    idx  = obj.systematic_resample(exp(logw));
                    X    = X(:, idx); logL = logL(idx); lp = lp(idx);
                    logw = -log(N) * ones(1, N);
                    obj.resample_count = obj.resample_count + 1;
                    [X, logL, lp, acc] = obj.mcmc_move(X, logL, lp, gamma, meas, logprior, fvar);
                    acc_sum = acc_sum + acc; acc_cnt = acc_cnt + 1;
                else
                    % [修复 γ=1 退出缺陷]：到达终点 γ=1 时，若经历了多步退火或最终 ESS 低于重采样阈值，
                    % 先用当前精确连续权重更新一次粒子状态，再重采样并在 γ=1 完整后验下做最终 MCMC 细化，
                    % 消除残留的低权重废粒子，确保进入下一拍预测步的粒子 100% 有效！
                    if n_stage > 1 || obj.ess_of(logw) < ess_res_tgt
                        idx  = obj.systematic_resample(exp(logw));
                        X    = X(:, idx); logL = logL(idx); lp = lp(idx);
                        logw = -log(N) * ones(1, N);
                        obj.resample_count = obj.resample_count + 1;
                        [X, logL, lp, acc] = obj.mcmc_move(X, logL, lp, 1.0, meas, logprior, fvar);
                        acc_sum = acc_sum + acc; acc_cnt = acc_cnt + 1;
                    end
                end
            end

            % 写回粒子状态并计算最终 MMSE 后验估计
            obj.P_p = X(1:nP, :);
            obj.P_v = X(nP+1:2*nP, :);
            obj.from_chart(X(2*nP+1:end, :), Rref);
            obj.P_logw = log(max(exp(logw), realmin));
            obj.update_mmse_estimate();

            obj.last_num_stages = n_stage;
            obj.last_accept     = acc_sum / max(acc_cnt, 1);
        end

        function dg = next_gamma(obj, logw, logL, r, tgt)
            if obj.ess_of(logw + r * logL) >= tgt
                dg = r; return;
            end
            lo = 0; hi = r;
            for it = 1:50
                mid = 0.5 * (lo + hi);
                if obj.ess_of(logw + mid * logL) >= tgt
                    lo = mid;
                else
                    hi = mid;
                end
            end
            dg = lo;
        end

        function [X, logL, lp, acc] = mcmc_move(obj, X, logL, lp, gamma, meas, logprior, fvar)
            I_num = obj.Vehicle_num;
            N  = size(X, 2);
            d  = size(X, 1);
            nP = 3 * I_num;

            Xc = X - mean(X, 2);
            Sc = (Xc * Xc') / (N - 1);
            Sc = (Sc + Sc') / 2 + diag(fvar);
            [Lc, flag] = chol(Sc, 'lower');
            if flag ~= 0, Lc = diag(sqrt(diag(Sc))); end

            acc_tot = 0;
            for it = 1:obj.n_mcmc
                Xp    = X + obj.mcmc_scale * (Lc * randn(d, N));
                logLp = obj.loglik(Xp(1:nP, :), meas);
                lpp   = logprior(Xp);
                log_a = gamma * (logLp - logL) + (lpp - lp);
                ok    = log(rand(1, N)) < log_a;

                X(:, ok) = Xp(:, ok);
                logL(ok) = logLp(ok);
                lp(ok)   = lpp(ok);

                a = mean(ok);
                acc_tot = acc_tot + a;
                obj.mcmc_scale = min(1.5, max(0.02, ...
                    obj.mcmc_scale * exp(a - obj.mcmc_acc_target)));
            end
            acc = acc_tot / obj.n_mcmc;
        end

        function fv = floor_variance(obj)
            n3 = 3 * obj.Vehicle_num;
            fv = [obj.min_pos_jitter^2 * ones(n3, 1); ...
                  obj.min_vel_jitter^2 * ones(n3, 1); ...
                  obj.min_att_jitter^2 * ones(n3, 1)] + 1e-12;
        end

        function e = ess_of(~, logw)
            w = exp(logw - max(logw));
            e = sum(w)^2 / sum(w.^2);
        end

        % ---------------- 旧的批处理路径 (对照用) ----------------
        function severe = update_batched(obj, meas)
            M  = numel(meas.val);
            bs = max(1, obj.batch_size);
            order = randperm(M);
            nb = ceil(M / bs);
            severe = false;
            for b = 1:nb
                sel = order((b-1)*bs+1 : min(b*bs, M));
                sub.val = meas.val(sel);   sub.type = meas.type(sel);
                sub.i   = meas.i(sel);     sub.j    = meas.j(sel);
                sub.var = meas.var(sel);

                lw = obj.P_logw + obj.loglik(obj.P_p, sub);
                lw = lw - max(lw);
                w  = exp(lw); w = w / sum(w);
                obj.P_logw = log(max(w, realmin));

                Neff = 1 / sum(w.^2);
                if Neff < obj.neff_ratio * obj.Np
                    if Neff / obj.Np < 0.05, severe = true; end
                    obj.resample_and_roughen(w);
                end
            end
            obj.update_mmse_estimate();
        end

        function resample_and_roughen(obj, w_norm)
            obj.resample_count = obj.resample_count + 1;
            N = obj.Np;

            std_p_pre = std(obj.P_p, 0, 2);
            std_v_pre = std(obj.P_v, 0, 2);

            idx = obj.systematic_resample(w_norm);
            obj.P_p    = obj.P_p(:, idx);
            obj.P_v    = obj.P_v(:, idx);
            obj.P_R    = obj.P_R(:, :, :, idx);
            obj.P_logw = log(1 / N) * ones(1, N);

            jit_p = min(0.2, obj.rough_coeff * std_p_pre + obj.min_pos_jitter);
            jit_v = min(0.2, obj.rough_coeff * std_v_pre + obj.min_vel_jitter);
            obj.P_p = obj.P_p + jit_p .* randn(size(obj.P_p));
            obj.P_v = obj.P_v + jit_v .* randn(size(obj.P_v));

            for vi = 1:obj.Vehicle_num
                Ri = reshape(obj.P_R(:, :, vi, :), 3, 3, N);
                [U, ~, V] = svd(sum(Ri, 3));
                R_mean = U * diag([1, 1, det(U * V')]) * V';
                err = obj.log_SO3_batch(pagemtimes(R_mean, 'transpose', Ri, 'none'));
                sigma_theta = sqrt(mean(sum(err.^2, 1)));
                jit_att = obj.rough_coeff * sigma_theta + obj.min_att_jitter;
                if jit_att > 0
                    Ri = pagemtimes(Ri, obj.exp_SO3_batch(jit_att * randn(3, N)));
                    obj.P_R(:, :, vi, :) = reshape(Ri, 3, 3, 1, N);
                end
            end
        end

        function idx = systematic_resample(~, w)
            w = w(:) / sum(w);
            N = numel(w);
            c = cumsum(w);
            u = ((0:N-1)' + rand()) / N;
            idx = zeros(1, N);
            i = 1;
            for j = 1:N
                while i < N && u(j) > c(i)
                    i = i + 1;
                end
                idx(j) = i;
            end
        end

        function update_mmse_estimate(obj)
            max_logw = max(obj.P_logw);
            w_norm   = exp(obj.P_logw - max_logw);
            w_norm   = w_norm(:) / sum(w_norm);

            obj.p = obj.P_p * w_norm;
            obj.v = obj.P_v * w_norm;

            for i = 1:obj.Vehicle_num
                R_i_all = reshape(obj.P_R(:, :, i, :), 9, obj.Np);
                R_bar   = reshape(R_i_all * w_norm, 3, 3);
                [U, ~, V] = svd(R_bar);
                obj.R(:, :, i) = U * diag([1, 1, det(U * V')]) * V';
            end
        end

        % =========================================================
        % SO(3) 切空间图表: delta = log(R_ref' * R_k)
        % =========================================================
        function [X, Rref] = to_chart(obj)
            I_num = obj.Vehicle_num;
            N  = obj.Np;
            nP = 3 * I_num;
            X  = zeros(9 * I_num, N);
            X(1:nP, :)        = obj.P_p;
            X(nP+1:2*nP, :)   = obj.P_v;
            Rref = obj.R;
            for i = 1:I_num
                Ri  = reshape(obj.P_R(:, :, i, :), 3, 3, N);
                rel = pagemtimes(Rref(:, :, i), 'transpose', Ri, 'none');
                X(2*nP+3*i-2 : 2*nP+3*i, :) = obj.log_SO3_batch(rel);
            end
        end

        function from_chart(obj, Phi, Rref)
            I_num = obj.Vehicle_num;
            N = obj.Np;
            for i = 1:I_num
                Rn = pagemtimes(repmat(Rref(:, :, i), 1, 1, N), ...
                                obj.exp_SO3_batch(Phi(3*i-2:3*i, :)));
                obj.P_R(:, :, i, :) = reshape(Rn, 3, 3, 1, N);
            end
        end

        % =========================================================
        % 李群/李代数工具
        % =========================================================
        function S = skew(~, v)
            S = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        end

        function v = log_SO3(~, R)
            tr = trace(R);
            theta = acos(max(-1, min(1, (tr - 1) / 2)));
            if theta < 1e-6
                v = zeros(3, 1);
            else
                w_skew = (theta / (2 * sin(theta))) * (R - R');
                v = [w_skew(3, 2); w_skew(1, 3); w_skew(2, 1)];
            end
        end

        function R = exp_SO3(obj, v)
            theta = norm(v);
            if theta < 1e-6
                R = eye(3) + obj.skew(v);
            else
                n = v / theta; S = obj.skew(n);
                R = eye(3) + sin(theta) * S + (1 - cos(theta)) * (S * S);
            end
        end

        function R = exp_SO3_batch(~, phi)
            N = size(phi, 2);
            x = phi(1, :); y = phi(2, :); z = phi(3, :);
            th2 = x.^2 + y.^2 + z.^2;
            th  = sqrt(th2);
            small = th < 1e-6;
            ths = th; ths(small) = 1;
            A = sin(ths) ./ ths;               A(small) = 1 - th2(small) / 6;
            B = (1 - cos(ths)) ./ ths.^2;      B(small) = 0.5 - th2(small) / 24;
            C = 1 - B .* th2;

            R = zeros(3, 3, N);
            R(1,1,:) = reshape(C + B.*x.^2,   1, 1, N);
            R(2,2,:) = reshape(C + B.*y.^2,   1, 1, N);
            R(3,3,:) = reshape(C + B.*z.^2,   1, 1, N);
            R(1,2,:) = reshape(B.*x.*y - A.*z, 1, 1, N);
            R(2,1,:) = reshape(B.*x.*y + A.*z, 1, 1, N);
            R(1,3,:) = reshape(B.*x.*z + A.*y, 1, 1, N);
            R(3,1,:) = reshape(B.*x.*z - A.*y, 1, 1, N);
            R(2,3,:) = reshape(B.*y.*z - A.*x, 1, 1, N);
            R(3,2,:) = reshape(B.*y.*z + A.*x, 1, 1, N);
        end

        function v = log_SO3_batch(~, R)
            N = size(R, 3);
            tr = reshape(R(1,1,:) + R(2,2,:) + R(3,3,:), 1, N);
            w = 0.5 * [reshape(R(3,2,:) - R(2,3,:), 1, N); ...
                       reshape(R(1,3,:) - R(3,1,:), 1, N); ...
                       reshape(R(2,1,:) - R(1,2,:), 1, N)];
            s = sqrt(sum(w.^2, 1));
            c = 0.5 * (tr - 1);
            theta = atan2(s, c);
            f = ones(1, N);
            big = s > 1e-9;
            f(big) = theta(big) ./ s(big);
            v = w .* f;
        end
    end
end