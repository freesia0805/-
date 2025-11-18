% gen_ofdm_dataset.m
rng(42);

%% 파라미터
FFT_N = 64;
CP = 16;
M = 4;              % QPSK
Fs = 20e6;
K_active = 52;      % DC/guard 제외한 유효 톤 수
frames_per_snr = 5000;
snr_list = [0, 10, 20];
isr_list = [-5, 0, 5];

out_folder = 'mat_dataset';
if ~exist(out_folder,'dir'), mkdir(out_folder); end

%% 활성 서브캐리어 인덱스 (1-based)
% 가운데 DC를 제외하고 좌우 K_active/2 씩 선택하는 단순 예시
all_k   = 1:FFT_N;
center  = FFT_N/2 + 1;                % 33
half    = floor(K_active/2);          % 26
cand    = [ (center-half):(center-1), (center+1):(center+half) ];
active_idx = cand;                    % 길이 == K_active, DC(33) 제외

%% 누적 버퍼
data_list  = [];   % (K_active, N) 방향으로 쌓고 마지막에 전치
label_list = [];
meta       = struct('snr', {});   % meta(end+1).snr = SNR

%% 프레임 생성
for s = 1:length(snr_list)
    SNR = snr_list(s);

    for f = 1:frames_per_snr
        % -------- 1) 랜덤 QPSK 심볼 생성 & 주파수 자원 매핑 --------
        bits = randi([0 M-1], K_active, 1);
        sym  = pskmod(bits, M, pi/M);             % QPSK (Gray 회전)
        Xfd  = zeros(FFT_N,1);
        Xfd(active_idx) = sym;

        % -------- 2) IFFT -> CP 삽입 --------
        tx_time = ifft(Xfd) * sqrt(FFT_N);
        tx_cp   = [tx_time(end-CP+1:end); tx_time];

        % -------- 3) 1-tap Rayleigh + AWGN(SNR) --------
        h  = (randn+1j*randn)/sqrt(2);
        rx = conv(tx_cp, h, 'same');

        sigP   = mean(abs(rx).^2);
        noiseP = sigP / (10^(SNR/10));
        rx = rx + sqrt(noiseP/2) * (randn(size(rx)) + 1j*randn(size(rx)));

        % -------- 4) 간섭 주입 --------
        % 두 유형 중 하나를 랜덤 선택
        isr_db = isr_list(randi(length(isr_list)));
        isr    = 10^(isr_db/10);

        % 협대역: 활성 톤 중 하나 k0 고르고, 그 주변에 약간 오프그리드 톤 주입(시간영역)
        % 광대역: FFT 후 선택 서브밴드에 잡음 주입(주파수영역) -> 라벨과 일치
        do_narrow = rand < 0.5;

        if do_narrow
            % ---- 협대역 톤 (time-domain) ----
            k0_pos = randi(length(active_idx));        % active 배열 내 위치
            k0     = active_idx(k0_pos);               % 실제 FFT bin (1..FFT_N)
            delta  = (rand-0.5) * 0.6 / FFT_N;         % 오프그리드 오프셋(±0.3 bin)
            w0     = 2*pi*((k0-1)/FFT_N + delta);      % 라디안 주파수
            n      = (0:length(rx)-1).';
            tone   = exp(1j*(w0*n + 2*pi*rand));       % 임의 위상

            toneP  = mean(abs(tone).^2);
            scale  = sqrt(isr * sigP / (toneP + 1e-12));
            rx     = rx + scale * tone;

            % 레이블: 중심 톤만 1 (원하면 아래처럼 ±1 dilate)
            lab_mask_fd = zeros(FFT_N,1);
            lab_mask_fd(k0) = 1;
            % % 옵션: 누설 고려하여 dilate
            % left  = active_idx(max(1,k0_pos-1));
            % right = active_idx(min(length(active_idx),k0_pos+1));
            % lab_mask_fd([left,right]) = 1;

            label_fd = lab_mask_fd;

            % FFT로 넘어가서 전력 계산
            rx_no_cp = rx(CP+1:end);
            Yfd = fft(rx_no_cp)/sqrt(FFT_N);

        else
            % ---- 광대역 (freq-domain) ----
            % 먼저 FFT
            rx_no_cp = rx(CP+1:end);
            Yfd = fft(rx_no_cp)/sqrt(FFT_N);

            % 활성 톤 내에서 연속 서브밴드 선택
            w = randi([4, 10]);                                % 대역폭(서브캐리어 수)
            start_pos = randi([1, length(active_idx)-w+1]);    % active 내 시작 위치
            band_pos  = start_pos:(start_pos+w-1);             % active 배열의 위치 인덱스
            band_bins = active_idx(band_pos);                  % 실제 FFT bin 인덱스

            % 선택 밴드에 잡음 주입 (주파수영역)
            band_noise_fd = (randn(w,1) + 1j*randn(w,1));
            % 스케일: 간섭 전력 = isr * 신호 전력(FFT 도메인에서 평균 사용)
            sigP_fd = mean(abs(Yfd).^2);
            scale   = sqrt(isr * sigP_fd / (mean(abs(band_noise_fd).^2) + 1e-12));
            Yfd(band_bins) = Yfd(band_bins) + scale * band_noise_fd;

            % 레이블: 선택 밴드 1
            label_fd = zeros(FFT_N,1);
            label_fd(band_bins) = 1;
        end

        % -------- 5) 출력 벡터 & 라벨 (활성 톤만 추출) --------
        P = abs(Yfd).^2;                  % 전력 스펙트럼
        data_list  = cat(2, data_list,  P(active_idx));           % (K_active, N)
        label_list = cat(2, label_list, label_fd(active_idx));    % (K_active, N)

        % -------- 6) 메타 정보 --------
        meta(end+1).snr = SNR;            %#ok<SAGROW>
    end
end

%% (N, K_active) 형태로 저장
X = data_list.';    % (N, K_active)
Y = label_list.';   % (N, K_active)

save(fullfile(out_folder,'ofdm_power_dataset.mat'), 'X', 'Y', 'active_idx', 'meta', '-v7.3');
disp('Saved: mat_dataset/ofdm_power_dataset.mat');

out_folder = 'mat_dataset_v2';
if ~exist(out_folder,'dir'), mkdir(out_folder); end
save(fullfile(out_folder,'ofdm_power_dataset_v2.mat'),'X','Y','active_idx','meta','-v7.3');
