        for this_gauge in self.ceh_gauge_id_no:
			print(this_gauge)
            # run method from super for each id
            [flux_dict[this_gauge], this_gauge_ll]  = fr.River.retrieveFlux(self, this_gauge, ceh_data_path=ceh_data_path)
            all_gauge_lon.append(this_gauge_ll[0])
            all_gauge_lat.append(this_gauge_ll[1])

