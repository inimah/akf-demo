# Muat konfigurasi lokal dari .env (dipanggil dengan: source muat_env.sh)
# Hanya mengisi variabel yang BELUM di-export, jadi nilai dari shell tetap diutamakan.
ENV_FILE=${ENV_FILE:-${SLURM_SUBMIT_DIR:-$(pwd)}/.env}
if [ -f "${ENV_FILE}" ]; then
    while IFS='=' read -r _k _v; do
        [[ "${_k}" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue          # lewati komentar/baris kosong
        _v=${_v%\"}; _v=${_v#\"}; _v=${_v%\'}; _v=${_v#\'}
        if [ -z "${!_k:-}" ]; then export "${_k}=${_v}"; fi
    done < "${ENV_FILE}"
    unset _k _v
else
    echo "Peringatan: ${ENV_FILE} tidak ditemukan (salin dari .env.example)." >&2
fi
