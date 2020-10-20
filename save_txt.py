def Save__File(self):
    # S_File will get the directory path and extension.
    S__File = QtWidgets.QFileDialog.getSaveFileName(None,'SaveTextFile','/', "Text Files (*.txt)")

    # This will let you access the test in your QTextEdit
    Text = self.Text__Edit.toPlainText()

    # This will prevent you from an error if pressed cancel on file dialog.
    if S__File[0]: 
       # Finally this will Save your file to the path selected.
       with open(S__File[0], 'w') as file:
       file.write(Text)