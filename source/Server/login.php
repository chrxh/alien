<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $success = checkPw($db, $userName, $pw);

    echo json_encode(["result"=>$success]);
    $db->close();
?>
