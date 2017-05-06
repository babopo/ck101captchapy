# xls test

import os
import xlrd
import xlwt
from xlutils.copy import copy
import extra


values = [1, 2, 3, 4]
book = xlrd.open_workbook("feature.xls")
br = copy(book)
a = book.sheet_by_index(0)
sh = br.get_sheet(0)  # sheet_by_index没有write方法
rows = a.nrows

for i in range(0, len(values)):
    sh.write(rows, i, values[i])

br.save("feature.xls")
#extra.xls_clear("feature.xls")
