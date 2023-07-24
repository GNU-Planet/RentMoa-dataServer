        plt.scatter(filtered_data['monthly_rent'], filtered_data['duration_months'], c=filtered_data['deposit'], cmap='viridis', s=filtered_data['deposit']/1000)
        plt.colorbar(label='보증금')
        plt.xlabel('월세')
        plt.ylabel('계약 기간')
        plt.title('보증금과 계약 기간에 따른 월세=0인 계약')
        plt.show()
