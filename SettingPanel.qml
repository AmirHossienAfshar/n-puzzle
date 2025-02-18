import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Pyside_Setting 1.0

Item {
    property alias trainingProgressValue: trainValue.value
    property alias startSolverEnabled: startSolverButton.enabled
    property alias generatePuzzleEnabled: generatePuzzleBtn.enabled

    id: settingsPanel
    Pyside_Setting_class {
        id: settings
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        GroupBox {
            id: puzzleSettings
            title: "Puzzle Settings"
            Layout.fillWidth: true
            Layout.fillHeight: true
            font.pixelSize: 14

            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                GridLayout {
                    columns: 2
                    rowSpacing: 5
                    columnSpacing: 10

                    Label {
                        text: "Puzzle grid size:"
                    }
                    ComboBox {
                        id: puzzleSizeComboBox
                        Layout.fillWidth: true
                        model: ["3x3", "4x4", "5x5", "6x6", "7x7", "8x8"]
                    }
                }
                Button {
                    id: generatePuzzleBtn
                    text: "Generate Puzzle!"
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
                    onClicked: {
                        let selectedSize = puzzleSizeComboBox.currentText;
                        settings.setting_set_puzzle_size(selectedSize)
                        settings.setting_initiate_generate_puzzle()
                    }
                }
            }
        }

        GroupBox {
            id: agentSettings
            title: "Agent Settings"
            Layout.fillWidth: true
            Layout.fillHeight: true
            font.pixelSize: 14

            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                GridLayout {
                    columns: 2
                    rowSpacing: 5
                    columnSpacing: 10

                    Label {
                        text: "Agent type:"
                    }
                    ComboBox {
                        id: agentComboBox
                        Layout.fillWidth: true
                        model: ["Q-Learning", "A*", "Sarsa", "row greedy"]
                    }
                    Label {
                        text: "Solver speed (step/sec):"
                    }
                    ComboBox {
                        id: solverSpeedComboBox
                        Layout.fillWidth: true
                        model: ["1", "2", "4"]
                    }
                    Label {
                        visible: agentComboBox.currentText === "Q-Learning" || agentComboBox.currentText === "Sarsa"
                        text: "Train episode number:"
                    }
                    // TextArea {
                    //     id: episodeNumberInput
                    //     Layout.fillWidth: true
                    //     text: "1000"
                    // }
                    TextField {
                        visible: agentComboBox.currentText === "Q-Learning" || agentComboBox.currentText === "Sarsa"
                        id: episodeNumberInput
                        Layout.fillWidth: true
                        text: "1000"
                        validator: IntValidator { bottom: 0; top: 100000 }
                        inputMethodHints: Qt.ImhDigitsOnly
                    }
                }
                Button {
                    visible: agentComboBox.currentText === "Q-Learning" || agentComboBox.currentText === "Sarsa"
                    text: "Train!"
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
                    onClicked: {
                        settings.setting_set_agent_type(agentComboBox.currentText)
                        settings.setting_set_solver_speed(solverSpeedComboBox.currentText)
                        settings.setting_set_episode_number(episodeNumberInput.text)
                    }
                }
            }
        }

        GroupBox {
            visible: agentComboBox.currentText === "Q-Learning" || agentComboBox.currentText === "Sarsa"
            id: trainingProgress
            title: "Training Progress"
            Layout.fillWidth: true
            Layout.fillHeight: false
            font.pixelSize: 14

            RowLayout {
                anchors.fill: parent
                spacing: 10

                Label {
                    text: "Training:"
                }
                ProgressBar {
                    id: trainValue
                    Layout.fillWidth: true
                    value: 0.0  // Example progress value
                }
            }
        }
        GroupBox {
            id: solverControl
            title: "Solver Control"
            Layout.fillWidth: true
            Layout.fillHeight: false
            font.pixelSize: 14

            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                Button {
                    id: startSolverButton
                    text: "Start the Solver"
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
                    enabled: agentComboBox.currentText === "A*" // this one has to be overrided with the properity alias. has problems...
                    onClicked: {
                        settings.setting_set_solver_speed(solverSpeedComboBox.currentText)
                        settings.setting_initiate_solve_puzzle()
                        
                    }
                }
            }
        }
    }
}
