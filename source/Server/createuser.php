<?php
    use PHPMailer\PHPMailer\PHPMailer;
    use PHPMailer\PHPMailer\Exception;

    require './helpers.php';
    require './PHPMailer/src/Exception.php';
    require './PHPMailer/src/PHPMailer.php';
    require './PHPMailer/src/SMTP.php';
    
    $db = connectToDB();
    $db->begin_transaction();

    $userName = $_POST["userName"];
    if (!preg_match("#^[^ ]+$#", $userName)) {
        echo json_encode(["result"=>false]);
        exit;
    }

    $pw = $_POST["password"];
    $email = str_replace(" ", "", $_POST["email"]);

    $salt = base64_encode(mcrypt_create_iv(16, MCRYPT_DEV_URANDOM));
    $pwHash = hash("sha256", $pw . $salt);
    $emailHash = hash("sha256", $email);
    $activationCode = bin2hex(mcrypt_create_iv(3, MCRYPT_DEV_URANDOM));

    $obj = $db->query(
        "SELECT 
            u.PW_HASH as pwHash,
            u.SALT as salt,
            u.ACTIVATION_CODE as activationCode
        FROM user u
        WHERE u.NAME='".addslashes($userName)."'")->fetch_object();
    if ($obj && $obj->activationCode != "") {
        if (!$db->query("DELETE FROM user WHERE NAME='".addslashes($userName)."'")) {
            echo json_encode(["result"=>false]);
            $db->close();
            exit;
        }
    }

    $success = false;
    if ($db->query("INSERT INTO user (ID, NAME, PW_HASH, EMAIL_HASH, SALT, ACTIVATION_CODE, FLAGS, TIMESTAMP) 
        VALUES (NULL, '".addslashes($userName)."', '".addslashes($pwHash)."', '".addslashes($emailHash)."', '".addslashes($salt)."', '$activationCode', 0, NULL)")) {
        $mail = new PHPMailer(true);                              // Passing `true` enables exceptions
        try {
            //Server settings
            //$mail->SMTPDebug = 2;                                 // Enable verbose debug output
            $mail->isSMTP();                                      // Set mailer to use SMTP
            $mail->Host = 'alfa3211.alfahosting-server.de';                  // Specify main and backup SMTP servers
            $mail->SMTPAuth = true;                               // Enable SMTP authentication
            $mail->Username = '[username]';             // SMTP username
            $mail->Password = '[password]';                           // SMTP password
            $mail->SMTPSecure = 'ssl';                            // Enable SSL encryption, TLS also accepted with port 465
            $mail->Port = 465;                                    // TCP port to connect to
        
            //Recipients
            $mail->setFrom('info@alien-project.org', 'User registration');          //This is the email your form sends From
            $mail->addAddress($email, ''); // Add a recipient address
            //$mail->addAddress('contact@example.com');               // Name is optional
            //$mail->addReplyTo('info@example.com', 'Information');
            //$mail->addCC('cc@example.com');
            //$mail->addBCC('bcc@example.com');
        
            //Attachments
            //$mail->addAttachment('/var/tmp/file.tar.gz');         // Add attachments
            //$mail->addAttachment('/tmp/image.jpg', 'new.jpg');    // Optional name
        
            //Content
            $mail->isHTML(true);                                  // Set email format to HTML
            $mail->Subject = "Artificial Life Environment: confirmation code for user '" . addslashes($userName) . "'";
            $mail->Body    = "Your confirmation code for user '".addslashes($userName)."' is:\n\n" . $activationCode;
            //$mail->AltBody = 'This is the body in plain text for non-HTML mail clients';
        
            $mail->send();
            $success = true;
        } catch (Exception $e) {
            $success = false;
        }
    }
    echo json_encode(["result"=>$success]);

    $db->commit();
    $db->close();
?>