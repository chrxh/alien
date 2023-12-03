<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    $pw = $_POST["password"];
    $success = checkPw($db, $userName, $pw);

    if ($success) {
        $success = $db->query("UPDATE user SET FLAGS=0, TIMESTAMP=CURRENT_TIMESTAMP WHERE NAME='".addslashes($userName)."'");
    }

    echo json_encode(["result"=>$success]);
    $db->commit();
    $db->close();
?>
