for i in $(seq 1 20);
do
    nvprof  --profile-from-start off --aggregate-mode off --csv --event-collection-mode continuous -m nvlink_total_data_transmitted,nvlink_total_data_received,nvlink_transmit_throughput,nvlink_receive_throughput ./p2p
done



