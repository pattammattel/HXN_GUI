def parse_scan_range(scan_range, scan_numbers, str_scan_range):

    slist = str_scan_range.split(",")
    for item in slist:
        if "-" in item:
            slist = item.split("-")
            scan_range.append((int(slist[0].strip()), int(slist[1].strip())))
        else:
            scan_numbers.append(int(item.strip()))

    return scan_range, scan_numbers